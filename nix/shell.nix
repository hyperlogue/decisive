{pkgs}:
pkgs.mkShell {
  name = "decisive-dev";

  packages = with pkgs; [
    rustc
    cargo
    rustfmt
    clippy
    cmake
    ninja
    pkg-config
    uv
    python313
    openblas
    llvmPackages.lld
  ];
}
