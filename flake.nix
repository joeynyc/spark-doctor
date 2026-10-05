{
  description = "Local diagnostic CLI for NVIDIA DGX Spark";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      overlays.default = final: prev: {
        spark-doctor = final.callPackage ./nix/package.nix { };
      };

      packages = forAllSystems (pkgs: rec {
        spark-doctor = pkgs.callPackage ./nix/package.nix { };
        default = spark-doctor;
      });

      checks = forAllSystems (pkgs: {
        package = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          inputsFrom = [ self.packages.${pkgs.stdenv.hostPlatform.system}.default ];
          packages = [ pkgs.python3Packages.pytest ];
        };
      });

      formatter = forAllSystems (pkgs: pkgs.nixfmt);
    };
}
