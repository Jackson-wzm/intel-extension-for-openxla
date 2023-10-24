# -*- Python -*-
"""SYCL autoconfiguration.
`sycl_configure` depends on the following environment variables:

  * HOST_CXX_COMPILER:  The host C++ compiler
  * HOST_C_COMPILER:    The host C compiler
  * PYTHON_LIB_PATH: The path to the python lib
"""

_HOST_CXX_COMPILER = "HOST_CXX_COMPILER"

_HOST_C_COMPILER = "HOST_C_COMPILER"

_SYCL_TOOLKIT_PATH = "SYCL_TOOLKIT_PATH"

_SYCL_COMPILER_VERSION = "SYCL_COMPILER_VERSION"

_ONEAPI_MKL_PATH = "ONEAPI_MKL_PATH"

_TF_NEED_MKL = "TF_NEED_MKL"

_AOT_CONFIG = "AOT_CONFIG"

_PYTHON_LIB_PATH = "PYTHON_LIB_PATH"

_PYTHON_LIB_DIR = "PYTHON_LIB_DIR"

_PYTHON_BIN_PATH = "PYTHON_BIN_PATH"

def _enable_sycl(repository_ctx):
    if "TF_NEED_SYCL" in repository_ctx.os.environ:
        enable_sycl = repository_ctx.os.environ["TF_NEED_SYCL"].strip()
        return enable_sycl == "1"
    return False

def _enable_mkl(repository_ctx):
    if _TF_NEED_MKL in repository_ctx.os.environ:
        enable_mkl = repository_ctx.os.environ[_TF_NEED_MKL].strip()
        return enable_mkl == "1"
    return False

def _enable_sycl_build(repository_ctx):
    return _SYCL_TOOLKIT_PATH in repository_ctx.os.environ

def auto_configure_fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("\n%sAuto-Configuration Error:%s %s\n" % (red, no_color, msg))

def find_c(repository_ctx):
    """Find host C compiler."""
    c_name = "gcc"
    if _HOST_C_COMPILER in repository_ctx.os.environ:
        c_name = repository_ctx.os.environ[_HOST_C_COMPILER].strip()
    if c_name.startswith("/"):
        return c_name
    c = repository_ctx.which(c_name)
    if c == None:
        fail("Cannot find C compiler, please correct your path.")
    return c

def find_cc(repository_ctx):
    """Find host C++ compiler."""
    cc_name = "g++"
    if _HOST_CXX_COMPILER in repository_ctx.os.environ:
        cc_name = repository_ctx.os.environ[_HOST_CXX_COMPILER].strip()
    if cc_name.startswith("/"):
        return cc_name
    cc = repository_ctx.which(cc_name)
    if cc == None:
        fail("Cannot find C++ compiler, please correct your path.")
    return cc

def find_sycl_root(repository_ctx):
    """Find DPC++ compiler."""
    sycl_name = ""
    if _SYCL_TOOLKIT_PATH in repository_ctx.os.environ:
        sycl_name = str(repository_ctx.path(repository_ctx.os.environ[_SYCL_TOOLKIT_PATH].strip()).realpath)
    if sycl_name.startswith("/"):
        return sycl_name
    fail("Cannot find DPC++ compiler, please correct your path")

def find_sycl_include_path(repository_ctx):
    """Find DPC++ compiler."""
    base_path = find_sycl_root(repository_ctx)
    bin_path = repository_ctx.path(base_path + "/" + "bin" + "/" + "icpx")
    icpx_extra = ""
    if not bin_path.exists:
        bin_path = repository_ctx.path(base_path + "/" + "bin" + "/" + "clang")
        if not bin_path.exists:
            fail("Cannot find DPC++ compiler, please correct your path")
    else:
        icpx_extra = "-fsycl"
    gcc_path = repository_ctx.path("/usr/bin/gcc")
    gcc_install_dir = repository_ctx.execute([gcc_path, "-print-libgcc-file-name"])
    gcc_install_dir_opt = "--gcc-install-dir=" + str(repository_ctx.path(gcc_install_dir.stdout.strip()).dirname)
    cmd_out = repository_ctx.execute([bin_path, icpx_extra, gcc_install_dir_opt, "-xc++", "-E", "-v", "/dev/null", "-o", "/dev/null"])
    outlist = cmd_out.stderr.split("\n")
    real_base_path = str(repository_ctx.path(base_path).realpath).strip()
    include_dirs = []
    for l in outlist:
        if l.startswith(" ") and l.strip().startswith("/") and str(repository_ctx.path(l.strip()).realpath) not in include_dirs:
            include_dirs.append(str(repository_ctx.path(l.strip()).realpath))
    return include_dirs

def get_sycl_version(repository_ctx):
    """Get DPC++ compiler version yyyymmdd"""
    default_version = "00000000"
    macro = "__INTEL_LLVM_COMPILER"
    version_file = "include/sycl/CL/sycl/version.hpp"
    base_path = find_sycl_root(repository_ctx)
    intel_llvm_macro = "00000000"
    compiler_bin_path = base_path + "/bin/icpx"
    compiler_macros = repository_ctx.execute([compiler_bin_path, "-dM", "-E", "-xc++", "/dev/null"])
    macro_list = compiler_macros.stdout.split("\n")
    for m in macro_list:
        result = m.strip().split(" ")
        if macro in result:
            intel_llvm_macro = result[-1]
    if intel_llvm_macro >= "20230000":
        version_file = "include/sycl/version.hpp"
    full_path = repository_ctx.path(base_path + "/" + version_file)
    if not full_path.exists:
        return default_version
    f = repository_ctx.read(full_path)
    lines = str(f).split("\n")
    for l in lines:
        if l.startswith("#define"):
            l_list = l.strip().split(" ")
            if (l_list[0] == "#define" and
                l_list[1] == "__SYCL_COMPILER_VERSION"):
                default_version = l_list[-1]
    return default_version

def find_mkl_path(repository_ctx):
    """Find MKL Path."""
    mkl_path = ""
    if _ONEAPI_MKL_PATH in repository_ctx.os.environ:
        mkl_path = repository_ctx.os.environ[_ONEAPI_MKL_PATH].strip()
    if mkl_path.startswith("/"):
        return mkl_path
    fail("Cannot find OneAPI MKL, please correct your path")

def find_aot_config(repository_ctx):
    """Find AOT config."""
    device_tmp = " -Xs \'-device {}\'"
    if _AOT_CONFIG in repository_ctx.os.environ:
        devices = repository_ctx.os.environ[_AOT_CONFIG].strip()
        device_list = []
        if devices:
            device_list = devices.split(",")
        else:
            return ""
        if device_list:
            # check for security purpose only here
            for d in device_list:
                if len(d) > 20:
                    fail("Invalid AOT target: {}".format(d))
            device_tmp = device_tmp.format(devices)
    return device_tmp

def find_python_lib(repository_ctx):
    """Returns python path."""
    if _PYTHON_LIB_PATH in repository_ctx.os.environ:
        return repository_ctx.os.environ[_PYTHON_LIB_PATH].strip()
    fail("Environment variable PYTHON_LIB_PATH was not specified re-run ./configure")

def _check_lib(repository_ctx, toolkit_path, lib):
    """Checks if lib exists under sycl_toolkit_path or fail if it doesn't.

    Args:
      repository_ctx: The repository context.
      toolkit_path: The toolkit directory containing the libraries.
      ib: The library to look for under toolkit_path.
    """
    lib_path = toolkit_path + "/" + lib
    if not repository_ctx.path(lib_path).exists:
        auto_configure_fail("Cannot find %s" % lib_path)

def _check_dir(repository_ctx, directory):
    """Checks whether the directory exists and fail if it does not.

    Args:
      repository_ctx: The repository context.
      directory: The directory to check the existence of.
    """
    if not repository_ctx.path(directory).exists:
        auto_configure_fail("Cannot find dir: %s" % directory)

def _tpl_path(repository_ctx, labelname):
    return repository_ctx.path(Label("//third_party/gpus/%s.tpl" % labelname))

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    repository_ctx.template(
        out,
        _tpl_path(repository_ctx, tpl),
        substitutions,
    )

_INC_DIR_MARKER_BEGIN = "#include <...>"

def _cxx_inc_convert(path):
    """Convert path returned by cc -E xc++ in a complete path."""
    path = path.strip()
    return path

def _normalize_include_path(repository_ctx, path):
    """Normalizes include paths before writing them to the crosstool.

      If path points inside the 'crosstool' folder of the repository, a relative
      path is returned.
      If path points outside the 'crosstool' folder, an absolute path is returned.
      """
    path = str(repository_ctx.path(path))
    crosstool_folder = str(repository_ctx.path(".").get_child("crosstool"))

    if path.startswith(crosstool_folder):
        # We drop the path to "$REPO/crosstool" and a trailing path separator.
        return "\"" + path[len(crosstool_folder) + 1:] + "\""
    return "\"" + path + "\""

def _get_cxx_inc_directories_impl(repository_ctx, cc, lang_is_cpp):
    """Compute the list of default C or C++ include directories."""
    if lang_is_cpp:
        lang = "c++"
    else:
        lang = "c"
    result = repository_ctx.execute([cc, "-E", "-x" + lang, "-", "-v"])
    index1 = result.stderr.find(_INC_DIR_MARKER_BEGIN)
    if index1 == -1:
        return []
    index1 = result.stderr.find("\n", index1)
    if index1 == -1:
        return []
    index2 = result.stderr.rfind("\n ")
    if index2 == -1 or index2 < index1:
        return []
    index2 = result.stderr.find("\n", index2 + 1)
    if index2 == -1:
        inc_dirs = result.stderr[index1 + 1:]
    else:
        inc_dirs = result.stderr[index1 + 1:index2].strip()

    return [
        _normalize_include_path(repository_ctx, _cxx_inc_convert(p))
        for p in inc_dirs.split("\n")
    ]

def get_cxx_inc_directories(repository_ctx, cc):
    """Compute the list of default C and C++ include directories."""

    # For some reason `clang -xc` sometimes returns include paths that are
    # different from the ones from `clang -xc++`. (Symlink and a dir)
    # So we run the compiler with both `-xc` and `-xc++` and merge resulting lists
    includes_cpp = _get_cxx_inc_directories_impl(repository_ctx, cc, True)
    includes_c = _get_cxx_inc_directories_impl(repository_ctx, cc, False)

    return includes_cpp + [
        inc
        for inc in includes_c
        if inc not in includes_cpp
    ]

_DUMMY_CROSSTOOL_BZL_FILE = """
def error_gpu_disabled():
  fail("ERROR: Building with --config=sycl but TensorFlow is not configured " +
       "to build with GPU support. Please re-run ./configure and enter 'Y' " +
       "at the prompt to build with GPU support.")

  native.genrule(
      name = "error_gen_crosstool",
      outs = ["CROSSTOOL"],
      cmd = "echo 'Should not be run.' && exit 1",
  )

  native.filegroup(
      name = "crosstool",
      srcs = [":CROSSTOOL"],
      output_licenses = ["unencumbered"],
  )
"""

_DUMMY_CROSSTOOL_BUILD_FILE = """
load("//crosstool:error_gpu_disabled.bzl", "error_gpu_disabled")

error_gpu_disabled()
"""

def _create_dummy_repository(repository_ctx):
    # Set up BUILD file for sycl/.
    _tpl(repository_ctx, "sycl:build_defs.bzl")
    _tpl(repository_ctx, "sycl:BUILD")

    # If sycl_configure is not configured to build with SYCL support, and the user
    # attempts to build with --config=sycl, add a dummy build rule to intercept
    # this and fail with an actionable error message.
    repository_ctx.file(
        "crosstool/error_gpu_disabled.bzl",
        _DUMMY_CROSSTOOL_BZL_FILE,
    )
    repository_ctx.file("crosstool/BUILD", _DUMMY_CROSSTOOL_BUILD_FILE)

    _tpl(
        repository_ctx,
        "sycl:build_defs.bzl",
        {
            "%{sycl_is_configured}": "False",
            "%{sycl_build_is_configured}": "False",
            "%{mkl_is_configured}": "False",
        },
    )

def _sycl_autoconf_imp(repository_ctx):
    """Implementation of the sycl_autoconf rule."""
    builtin_include_dirs = ""
    unfiltered_cxx_flags = ""
    linker_flags = ""

    sycl_defines = {}

    if not _enable_sycl(repository_ctx):
        _create_dummy_repository(repository_ctx)
    else:
        tpl_paths = {labelname: _tpl_path(repository_ctx, labelname) for labelname in [
            # "rocm:build_defs.bzl",
            # "rocm:BUILD",
            "crosstool:BUILD.sycl",
            "crosstool:sycl_cc_toolchain_config.bzl",
            "crosstool/bin:crosstool_wrapper_driver",
            # "rocm:rocm_config.h",
        ]}

        # copy template files
        _tpl(repository_ctx, "sycl:build_defs.bzl")
        _tpl(repository_ctx, "sycl:BUILD")

        additional_cxxflags = []
        additional_linker_flags = []
        builtin_includes = []

        builtin_includes += [find_sycl_root(repository_ctx) + "/include"]
        builtin_includes += [find_sycl_root(repository_ctx) + "/lib/clang/12.0.0/include"]
        builtin_includes += [find_sycl_root(repository_ctx) + "/lib/clang/13.0.0/include"]

        pwd = repository_ctx.os.environ["PWD"]
        additional_inc = []
        if repository_ctx.os.environ.get("CPATH") != None:
            for p in repository_ctx.os.environ["CPATH"].strip().split(":"):
                if p != "":
                    additional_inc += [_normalize_include_path(repository_ctx, p)]
        if len(additional_inc) > 0:
            additional_inc = ",".join(additional_inc)
        else:
            additional_inc = "\"\""

        if _enable_mkl(repository_ctx) and repository_ctx.os.environ.get("ONEAPI_MKL_PATH") != None:
            sycl_defines["%{ONEAPI_MKL_PATH}"] = str(find_mkl_path(repository_ctx))
            builtin_includes += [find_mkl_path(repository_ctx) + "/include"]
        else:
            sycl_defines["%{ONEAPI_MKL_PATH}"] = ""
        if repository_ctx.os.environ.get("TMPDIR") != None:
            sycl_defines["%{TMP_DIRECTORY}"] = repository_ctx.os.environ.get("TMPDIR")
        else:
            tmp_suffix = repository_ctx.execute(["cat", "/proc/sys/kernel/random/uuid"]).stdout.rstrip()
            tmp_dir = "/tmp/" + tmp_suffix
            sycl_defines["%{TMP_DIRECTORY}"] = tmp_dir

        sycl_defines["%{cxx_builtin_include_directories}"] = str(builtin_includes)
        sycl_defines["%{sycl_builtin_include_directories}"] = str(builtin_includes)
        sycl_defines["%{extra_no_canonical_prefixes_flags}"] = "\"-fno-canonical-system-headers\""
        sycl_defines["%{unfiltered_compile_flags}"] = ""
        sycl_defines["%{host_compiler}"] = "gcc"
        sycl_defines["%{HOST_COMPILER_PATH}"] = "/usr/bin/gcc"
        sycl_defines["%{host_compiler_prefix}"] = "/usr/bin"
        sycl_defines["%{sycl_compiler_root}"] = str(find_sycl_root(repository_ctx))
        sycl_defines["%{linker_bin_path}"] = "/usr/bin"
        sycl_defines["%{SYCL_ROOT_DIR}"] = str(find_sycl_root(repository_ctx))
        sycl_defines["%{AOT_DEVICES}"] = str(find_aot_config(repository_ctx))
        sycl_defines["%{TF_NEED_MKL}"] = repository_ctx.os.environ[_TF_NEED_MKL].strip()
        sycl_defines["%{additional_include_directories}"] = additional_inc
        sycl_defines["%{SYCL_COMPILER_VERSION}"] = str(get_sycl_version(repository_ctx))
        sycl_defines["%{PYTHON_LIB_PATH}"] = repository_ctx.os.environ[_PYTHON_LIB_PATH]

        sycl_internal_inc_dirs = find_sycl_include_path(repository_ctx)
        sycl_internal_inc = "\", \"".join(sycl_internal_inc_dirs)
        sycl_internal_isystem_inc = []
        for d in sycl_internal_inc_dirs:
            sycl_internal_isystem_inc.append("-isystem\", \"" + d)

        if len(sycl_internal_inc_dirs) > 0:
            sycl_defines["%{SYCL_ISYSTEM_INC}"] = "\"]), \n\tflag_group(flags=[ \"".join(sycl_internal_isystem_inc)
            sycl_defines["%{SYCL_INTERNAL_INC}"] = sycl_internal_inc
        else:
            sycl_defines["%{SYCL_ISYSTEM_INC}"] = ""
            sycl_defines["%{SYCL_INTERNAL_INC}"] = ""

        unfiltered_cxx_flags = "" if additional_cxxflags == [] else "unfiltered_cxx_flag: "
        unfiltered_cxx_flags += "\n  unfiltered_cxx_flag: ".join(additional_cxxflags)

        sycl_defines["%{unfiltered_compile_flags}"] = unfiltered_cxx_flags

        linker_flags = "" if additional_linker_flags == [] else "linker_flag: "
        linker_flags += "\n  linker_flag: ".join(additional_linker_flags)

        # Only expand template variables in the BUILD file
        repository_ctx.template(
            "crosstool/BUILD",
            tpl_paths["crosstool:BUILD.sycl"],
            sycl_defines,
        )

        # No templating of cc_toolchain_config - use attributes and templatize the
        # BUILD file.
        repository_ctx.template(
            "crosstool/cc_toolchain_config.bzl",
            tpl_paths["crosstool:sycl_cc_toolchain_config.bzl"],
            sycl_defines,
        )

        repository_ctx.template(
            "crosstool/bin/crosstool_wrapper_driver",
            tpl_paths["crosstool/bin:crosstool_wrapper_driver"],
            sycl_defines,
        )

        if _enable_sycl_build(repository_ctx):
            sycl_build_defines = {}
            sycl_build_defines["%{sycl_is_configured}"] = "True"
            sycl_build_defines["%{sycl_build_is_configured}"] = "True"
            if _enable_mkl(repository_ctx):
                sycl_build_defines["%{mkl_is_configured}"] = "True"
            sycl_root = find_sycl_root(repository_ctx)
            _check_dir(repository_ctx, sycl_root)

            _tpl(
                repository_ctx,
                "sycl:build_defs.bzl",
                sycl_build_defines,
            )

sycl_configure = repository_rule(
    local = True,
    implementation = _sycl_autoconf_imp,
)
