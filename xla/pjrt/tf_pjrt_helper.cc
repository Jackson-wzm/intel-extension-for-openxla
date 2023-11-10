#include "xla/pjrt/tf_pjrt_helper.h"

#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tf_xpu_pjrt_client.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
xla::PrimitiveType XlaDataTypeFromString(std::string data_type) {
  if (data_type == "bool")
    return xla::PRED;
  else if (data_type == "int8" || data_type == "qint8")
    return xla::S8;
  else if (data_type == "int16" || data_type == "qint16")
    return xla::S16;
  else if (data_type == "int32" || data_type == "qint32")
    return xla::S32;
  else if (data_type == "int64")
    return xla::S64;
  else if (data_type == "uint8" || data_type == "quint8")
    return xla::U8;
  else if (data_type == "uint16" || data_type == "quint16")
    return xla::U16;
  else if (data_type == "uint32")
    return xla::U32;
  else if (data_type == "uint64")
    return xla::U64;
  else if (data_type == "bfloat16")
    return xla::BF16;
  else if (data_type == "half")
    return xla::F16;
  else if (data_type == "float")
    return xla::F32;
  else if (data_type == "double")
    return xla::F64;
  else if (data_type == "complex64")
    return xla::C64;
  else if (data_type == "complex128")
    return xla::C128;
  else
    return xla::PRIMITIVE_TYPE_INVALID;
}

void* ITEXOpaqueDataPointerFromPjRtBuffer(PJRT_Buffer* pjrt_c_buffer) {
  std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference_hold;
  external_reference_hold =
      std::move(pjrt_c_buffer->buffer->AcquireExternalReference().value());
  return external_reference_hold->OpaqueDeviceMemoryDataPointer();
}

PJRT_Buffer* ITEXCreatePjRtBuffer(int device_id, std::string data_type,
                                  std::vector<int64_t> dimentions,
                                  std::vector<int64_t> layout,
                                  PJRT_Client* pjrt_c_client) {
  xla::PjRtDevice* pjrt_device =
      pjrt_c_client->client->LookupDevice(device_id).value();
  xla::PrimitiveType type = XlaDataTypeFromString(data_type);
  xla::Shape shape =
      xla::ShapeUtil::MakeShapeWithDenseLayout(type, dimentions, layout);
  return new PJRT_Buffer{
      std::move(
          pjrt_c_client->client->CreateUninitializedBuffer(shape, pjrt_device)
              .value()),
      pjrt_c_client};
}

PJRT_Buffer* ITEXCopyFromPjRtBuffer(PJRT_Buffer* src, int device_id,
                                    std::string data_type,
                                    std::vector<int64_t> dimentions,
                                    std::vector<int64_t> layout,
                                    PJRT_Client* pjrt_c_client) {
  xla::PjRtDevice* pjrt_device =
      pjrt_c_client->client->LookupDevice(device_id).value();
  xla::PrimitiveType type = XlaDataTypeFromString(data_type);
  xla::Shape shape =
      xla::ShapeUtil::MakeShapeWithDenseLayout(type, dimentions, layout);
  auto src_scoped_hold =
      reinterpret_cast<xla::PjRtStreamExecutorBuffer*>(src->buffer.get())
          ->GetBufferWithExternalReference();
  auto src_tracked_buffer = src_scoped_hold.buffer();
  auto delete_callback = [src_tracked_buffer]() {
    auto buffer = const_cast<std::shared_ptr<xla::TrackedDeviceBuffer>&>(
        src_tracked_buffer);
    buffer.reset();
  };
  auto device_ptr = src_tracked_buffer->device_memory().front().opaque();
  auto pjrt_client = reinterpret_cast<xla::PjRtStreamExecutorClient*>(
      pjrt_c_client->client.get());
  return new PJRT_Buffer{
      std::move(pjrt_client
                    ->CreateViewOfDeviceBuffer(device_ptr, shape, pjrt_device,
                                               delete_callback, {})
                    .value()),
      pjrt_c_client};
}

void* ITEXGetStreamFromPjRtDevice(int device_id, PJRT_Client* pjrt_c_client) {
  xla::PjRtDevice* pjrt_device =
      std::move(pjrt_c_client->client->LookupDevice(device_id).value());
  void* stream = static_cast<void*>(stream_executor::gpu::AsGpuStreamValue(
      (static_cast<xla::PjRtStreamExecutorDevice*>(pjrt_device))
          ->local_device_state()
          ->compute_stream()));
  return stream;
}

namespace xla {

// Ensures that it is safe to deallocate any buffers that have been enqueued in
// an operation on stream. Called only in rare error cases that are triggered
// during enqueue. These cases generally correspond to resource exhaustion.
void StallStreamOnError(LocalDeviceState* local_device, se::Stream* stream) {
  switch (local_device->allocation_model()) {
    case LocalDeviceState::kAsynchronous:
      // We can safely deallocate any dangling buffers immediately. NOTE: this
      // assumes that any buffers enqueued on stream are local to stream's
      // executor, and manual action may be needed if that condition is not met.
      break;

    case LocalDeviceState::kComputeSynchronized:
      // This will stall computation but that's ok in this very rare error
      // case.
      if (stream != local_device->compute_stream()) {
        local_device->compute_stream()->ThenWaitFor(stream);
      }
      break;

    case LocalDeviceState::kSynchronous:
      // This will stall the calling thread but that's ok in this very rare
      // error case. If the stall fails just crash, since we have no other
      // way to synchronize.
      TF_CHECK_OK(stream->BlockHostUntilDone());
      break;
  }
}

// Adds necessary synchronization after a copy has been enqueued to a buffer.
// definition_event was added when the buffer was allocated, but has not yet
// had an event recorded.
Status AddDestinationBufferSynchronization(
    LocalDeviceState* local_device,
    std::shared_ptr<BufferSequencingEvent> definition_event,
    se::Stream* copy_stream) {
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(copy_stream);
  if (!event_or.ok()) {
    StallStreamOnError(local_device, copy_stream);
    return event_or.status();
  }
  definition_event->SetSequencingEvent(std::move(event_or).value(),
                                       copy_stream);
  return OkStatus();
}

PJRT_Buffer* SameDevicePjRtBufferCopy(PJRT_Buffer* src_buffer,
                                      PJRT_Client* c_client) {
  PjRtStreamExecutorBuffer* se_src_buffer =
      static_cast<PjRtStreamExecutorBuffer*>(src_buffer->buffer.get());
  if (se_src_buffer->on_device_shape().IsTuple()) {
    std::cout << "ITEXSameDevicePjRtBufferCopy does not support Tupple yet"
              << std::endl;
    std::abort();
  }

  PjRtStreamExecutorDevice* pjrt_device = se_src_buffer->device();
  LocalDeviceState* transfer_local_device = pjrt_device->local_device_state();

  se::Stream* transfer_stream =
      transfer_local_device->GetDeviceToDeviceStream();

  auto* se_client =
      static_cast<PjRtStreamExecutorClient*>(c_client->client.get());
  TransferManager* transfer_manager =
      se_client->client()->backend().transfer_manager();

  ScopedShapedBuffer dst_buffer =
      transfer_manager
          ->AllocateScopedShapedBuffer(se_src_buffer->on_device_shape(),
                                       se_client->allocator(),
                                       transfer_local_device->device_ordinal())
          .value();
  transfer_stream->ThenWaitFor(transfer_local_device->compute_stream());

  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 2>
      definition_events;
  definition_events.emplace_back(
      std::make_shared<BufferSequencingEvent>(se_client->thread_pool()));

  std::shared_ptr<TrackedDeviceBuffer> dst_device_buffer =
      TrackedDeviceBuffer::FromScopedShapedBuffer(&dst_buffer,
                                                  definition_events);
  auto py_dst_buffer = std::make_unique<PjRtStreamExecutorBuffer>(
      se_src_buffer->on_device_shape(), std::move(dst_device_buffer), se_client,
      pjrt_device);

  ShapedBuffer shaped_dst_buffer = py_dst_buffer->AsShapedBuffer().value();

  PjRtStreamExecutorBuffer::ScopedHold scoped_dst_buffer(
      py_dst_buffer->GetBufferWithUsageHold());
  // Copy the leaf buffers.
  ShapedBuffer shaped_src_buffer = se_src_buffer->AsShapedBuffer().value();

  StatusOr<std::shared_ptr<BufferSequencingEvent>> copy_event_or =
      [&]() -> StatusOr<std::shared_ptr<BufferSequencingEvent>> {
    for (const auto& leaf : shaped_src_buffer.buffers().leaves()) {
      const ShapeIndex& index = leaf.first;
      const se::DeviceMemoryBase& input_buffer = leaf.second;
      const se::DeviceMemoryBase& output_buffer =
          shaped_dst_buffer.buffer(index);
      TF_RET_CHECK(input_buffer.size() == output_buffer.size())
          << "input: " << input_buffer.size()
          << " output: " << output_buffer.size();
      if (input_buffer.size() != 0) {
        TF_RETURN_IF_ERROR(transfer_local_device->ThenMemcpyDeviceToDevice(
            transfer_stream, transfer_local_device->compute_stream(),
            input_buffer, output_buffer));
      }
    }
    std::shared_ptr<BufferSequencingEvent> event =
        scoped_dst_buffer->definition_events()[0];
    TF_RETURN_IF_ERROR(AddDestinationBufferSynchronization(
        transfer_local_device, event, transfer_stream));
    return event;
  }();

  if (!copy_event_or.ok()) {
    StallStreamOnError(transfer_local_device, transfer_stream);
  }

  return new PJRT_Buffer{std::move(py_dst_buffer), c_client};
}

}  // namespace xla

PJRT_Buffer* ITEXSameDevicePjRtBufferCopy(PJRT_Buffer* src_buffer,
                                          PJRT_Client* c_client) {
  return xla::SameDevicePjRtBufferCopy(src_buffer, c_client);
}

void ITEXXlaShapeToDeviceShapeRepresentation(void* serialized_xla_shape,
                                             void* serialized_device_shape) {
  xla::Shape xla_shape =
      ApiConverter::FromC(static_cast<XLA_Shape*>(serialized_xla_shape));
  ApiConverter::ToC(xla_shape,
                    static_cast<XLA_Shape*>(serialized_device_shape));
}
