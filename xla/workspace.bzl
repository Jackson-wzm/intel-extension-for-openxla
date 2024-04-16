load("//third_party/build_option:dpcpp_configure.bzl", "dpcpp_configure")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")

def workspace(path_prefix = "", tf_repo_name = ""):
    dpcpp_configure(name = "local_config_dpcpp")

    git_repository(
        name = "spir_headers",
        commit = "9b527c0fb60124936d0906d44803bec51a0200fb",
        remote = "https://github.com/KhronosGroup/SPIRV-Headers.git",
        verbose = True,
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )

    new_git_repository(
        name = "llvm_spir",
        commit = "d021f7535060ab7fc548a1446036c171f490d311",
        remote = "https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git",
        build_file = "//third_party/llvm_spir:llvm_spir.BUILD",
        verbose = True,
        patches = ["//third_party/llvm_spir:llvm_spir.patch"],
        patch_args = ["-p1"],
    )

    new_git_repository(
        name = "xetla",
        # v0.3.7.2
        commit = "ae46a690bac364a93437e248418636c2a8423134",
        remote = "https://github.com/intel/xetla.git",
        verbose = True,
        build_file = "//third_party/xetla:BUILD",
        patch_cmds = [
            "git log -1 --format=%H > COMMIT",
        ],
    )
