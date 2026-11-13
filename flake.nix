{
  description = "GraphIV";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/34ca302a9572963c02e385c056be37c85ff51b77";
  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
    lib = nixpkgs.lib;

    # Postgres 17 with pgvector — the SINGLE project store (metadata + citations + embeddings +
    # topic tree). Launched from THIS flake (not the global systemd service): a user-owned cluster
    # over a PROJECT-LOCAL unix socket, so it needs no password (peer auth) and never clashes with
    # any global postgres on :5432.
    pgWithVec = pkgs.postgresql_17.withPackages (p: [ p.pgvector ]);

    # Project-owned cluster, fully self-contained under ./data/pg (real dir, no symlink) on the
    # current NVMe. Socket + port are project-local so it never clashes with a global postgres.
    pgEnv = ''
      export PGDATA="''${PGDATA:-$PWD/data/pg}"
      export PGHOST="''${PGHOST:-$PWD/.pg}"
      export PGPORT="''${PGPORT:-5433}"
      export PGDATABASE="''${PGDATABASE:-arxivkg}"
    '';

    pg-init = pkgs.writeShellScriptBin "pg-init" ''
      set -euo pipefail
      ${pgEnv}
      if [ -d "$PGDATA/base" ]; then echo "cluster already initialized at $PGDATA"; exit 0; fi
      mkdir -p "$PGDATA" "$PGHOST"
      ${pgWithVec}/bin/initdb -D "$PGDATA" --encoding=UTF8 --no-locale -U "$USER"
      {
        echo "unix_socket_directories = '$PGHOST'"
        echo "listen_addresses = '''"                 # socket-only; no TCP exposure
        echo "port = $PGPORT"
        echo "default_toast_compression = lz4"        # compress in-DB text (sentence, etc.)
        echo "shared_buffers = 4GB"
        echo "work_mem = 256MB"
        echo "maintenance_work_mem = 2GB"             # speeds HNSW + index builds
        echo "max_parallel_maintenance_workers = 4"
      } >> "$PGDATA/postgresql.conf"
      echo "initialized. start with: pg-start"
    '';

    pg-start = pkgs.writeShellScriptBin "pg-start" ''
      set -euo pipefail
      ${pgEnv}
      ${pgWithVec}/bin/pg_ctl -D "$PGDATA" -l "$PGDATA/server.log" start
      # create the database + schema on first start (idempotent)
      ${pgWithVec}/bin/createdb -h "$PGHOST" -p "$PGPORT" "$PGDATABASE" 2>/dev/null || true
      ${pgWithVec}/bin/psql -h "$PGHOST" -p "$PGPORT" -d "$PGDATABASE" -v ON_ERROR_STOP=1 \
        -f "$PWD/scripts/papers_preprocess/schema.sql" \
        -f "$PWD/scripts/topic_tree_creation/schema.sql"
      echo "up: host=$PGHOST port=$PGPORT db=$PGDATABASE"
    '';

    pg-stop = pkgs.writeShellScriptBin "pg-stop" ''
      set -euo pipefail
      ${pgEnv}
      ${pgWithVec}/bin/pg_ctl -D "$PGDATA" stop -m fast
    '';

    pg-console = pkgs.writeShellScriptBin "pg-console" ''
      set -euo pipefail
      ${pgEnv}
      exec ${pgWithVec}/bin/psql -h "$PGHOST" -p "$PGPORT" -d "$PGDATABASE" "$@"
    '';
    overlay = final: _: {
      graphiv = final.callPackage ./nix/package.nix { };
    };
    graphiv = pkgs.callPackage ./nix/package.nix { };

    # the module on a bare host, enough to read its options and the config it writes
    probe = lib.nixosSystem {
      inherit system;
      modules = [
        self.nixosModules.default
        {
          boot.isContainer = true;
          system.stateVersion = "25.11";
          services.graphiv.enable = true;
          services.graphiv.settings.endpoints.LLM_NAME = "probe";
        }
      ];
    };
    keysOf = prefix: tables: lib.concatLists (lib.mapAttrsToList (k: v:
      if builtins.isAttrs v then keysOf (prefix ++ [ k ]) v else [ (lib.concatStringsSep "." (prefix ++ [ k ])) ]
    ) tables);
    declared = keysOf [ ] probe.config.services.graphiv.settings;
    shipped = keysOf [ ] (lib.importTOML ./config.toml);
    undeclared = lib.subtractLists declared shipped;
    unshipped = lib.subtractLists shipped declared;
  in {
    packages.${system} = {
      default = graphiv;
      inherit graphiv;
      # what the manager's config tab renders, for a manager run from the checkout
      config-schema = pkgs.runCommand "graphiv-config-schema.json" { } ''
        cp ${probe.config.systemd.services.graphiv-manager.environment.GRAPHIV_CONFIG_SCHEMA} $out
      '';
    };

    # every key config.toml ships is a module option, and every option a key it ships
    checks.${system}.config-keys = pkgs.runCommand "graphiv-config-keys" { } (
      if undeclared == [ ] && unshipped == [ ] then "touch $out"
      else "echo 'not a module option: ${toString undeclared}'; echo 'not in config.toml: ${toString unshipped}'; exit 1"
    );

    overlays.default = overlay;

    nixosModules.default = { ... }: {
      imports = [ ./nix/module.nix ];
      nixpkgs.overlays = [ overlay ];
    };

    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        graphiv.python313
        graphiv.uv
        ruff
        black
        pyright
        cudaPackages.cuda_cudart
        cudaPackages.libcublas
        cudaPackages.cuda_cccl
        git
        pgWithVec
        pg-init
        pg-start
        pg-stop
        pg-console
        # LaTeX for the experiment reports, the paper and the slides.
        # scheme-medium lacks tcolorbox, which the beamer theme requires.
        (texlive.combine {
          inherit (texlive)
            scheme-medium
            tcolorbox tikzfill pgf environ trimspaces pdfcol
            unicode-math ragged2e etoolbox kvoptions
            beamer translator;
        })
      ];
      DEV_SHELL = "GraphIV";
      # the same interpreter and libraries the daemons run on, so .venv and the deployed venv match
      UV_PYTHON = "${graphiv.python313}/bin/python3.13";
      UV_PYTHON_DOWNLOADS = "never";
      LD_LIBRARY_PATH = graphiv.libraryPath + ":" + lib.makeLibraryPath (with pkgs; [ cudaPackages.cuda_cudart cudaPackages.libcublas cudaPackages.cuda_cccl ]);
      shellHook = ''
        export PGHOST="''${PGHOST:-$PWD/.pg}"
        export PGPORT="''${PGPORT:-5433}"
        export PGDATABASE="''${PGDATABASE:-arxivkg}"
        export ARXIVKG_PG_DSN="''${ARXIVKG_PG_DSN:-host=$PWD/.pg port=5433 dbname=arxivkg}"
        if [ -d .venv ]; then
          export VIRTUAL_ENV="$PWD/.venv"
          export PATH="$PWD/.venv/bin:$PATH"
        fi
        [[ $- == *i* ]] && exec zsh
      '';
    };

    apps.${system} = {
      pg-init    = { type = "app"; program = "${pg-init}/bin/pg-init"; };
      pg-start   = { type = "app"; program = "${pg-start}/bin/pg-start"; };
      pg-stop    = { type = "app"; program = "${pg-stop}/bin/pg-stop"; };
      pg-console = { type = "app"; program = "${pg-console}/bin/pg-console"; };
    };
  };
}
