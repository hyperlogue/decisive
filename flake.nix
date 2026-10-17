{
  description = "Decisive — a fast decision-model inference engine for browsers and edge devices";

  inputs.nixpkgs.url = "https://releases.nixos.org/nixos/unstable/nixos-26.11pre1027867.d407951447dc/nixexprs.tar.xz";

  outputs = {nixpkgs, ...}: {
    devShells = nixpkgs.lib.genAttrs [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ] (system: {
      default = import ./nix/shell.nix {
        pkgs = nixpkgs.legacyPackages.${system};
      };
    });
  };
}
