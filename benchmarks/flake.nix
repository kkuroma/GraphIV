{
  description = "GraphIV baselines, in an environment of their own";
  # the same pin as the root flake, so both shells share one interpreter and one set of native libraries
  inputs.nixpkgs.url = "github:nixos/nixpkgs/34ca302a9572963c02e385c056be37c85ff51b77";
  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
    lib = nixpkgs.lib;
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [ python313 uv ];
      DEV_SHELL = "GraphIV-benchmarks";
      UV_PYTHON = "${pkgs.python313}/bin/python3.13";
      UV_PYTHON_DOWNLOADS = "never";
      # what the torch and tokenizer wheels load that NixOS keeps elsewhere, as in nix/package.nix
      LD_LIBRARY_PATH = "/run/opengl-driver/lib:" + lib.makeLibraryPath (with pkgs; [ stdenv.cc.cc.lib zlib libGL ]);
      shellHook = ''
        if [ -d .venv ]; then
          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$PWD/.venv/bin:$PATH"
        fi
      '';
    };
  };
}
