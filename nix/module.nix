# The NixOS module: the cluster, the venv, three listeners over one package, and config.toml
# settings is config.toml key for key, so the whole deployment is one services.graphiv block
{
  config,
  lib,
  pkgs,
  options,
  ...
}:

let
  inherit (lib) mkOption types;
  cfg = config.services.graphiv;
  pg = cfg.postgres;
  s = cfg.settings;
  toml = pkgs.formats.toml { };

  opt =
    type: default: description:
    mkOption { inherit type default description; };

  # a default read off another option, which the manual shows as text rather than evaluating
  derived =
    type: default: text: description:
    mkOption {
      inherit type default description;
      defaultText = lib.literalExpression text;
    };

  section =
    description: options:
    mkOption {
      inherit description;
      default = { };
      type = types.submodule { inherit options; };
    };

  file = toml.generate "graphiv-config.toml" s;

  # every section and key with its kind and description, which the manager's config tab renders
  kinds = {
    str = "str";
    int = "int";
    unsignedInt16 = "int";
    float = "float";
    bool = "bool";
    listOf = "list";
  };
  describe =
    prefix: opts:
    let
      inner = lib.filterAttrs (n: _: n != "_module") (opts.type.getSubOptions [ ]);
      plain = lib.filterAttrs (_: o: o.type.name != "submodule") inner;
      nested = lib.filterAttrs (_: o: o.type.name == "submodule") inner;
      name = lib.concatStringsSep "." prefix;
    in
    lib.optional (prefix != [ ]) {
      inherit name;
      description = opts.description or "";
      keys = lib.mapAttrsToList (
        key: o:
        {
          inherit key;
          kind = kinds.${o.type.name};
          description = lib.removeSuffix "\n" (o.description or "");
        }
        // lib.optionalAttrs (o.type.name == "listOf") { item = kinds.${o.type.nestedTypes.elemType.name}; }
      ) plain;
    }
    ++ lib.concatLists (lib.mapAttrsToList (n: o: describe (prefix ++ [ n ]) o) nested);
  # attribute sets are sorted, so the tab takes its order from the lines of the shipped config.toml
  order =
    let
      step =
        acc: line:
        let
          header = builtins.match "\\[([A-Za-z_.]+)].*" line;
          entry = builtins.match "([A-Za-z_]+) *=.*" line;
        in
        if header != null then
          {
            section = builtins.head header;
            out = acc.out ++ header;
          }
        else if entry != null then
          acc // { out = acc.out ++ [ "${acc.section}.${builtins.head entry}" ]; }
        else
          acc;
    in
    (builtins.foldl' step {
      section = "";
      out = [ ];
    } (lib.splitString "\n" (builtins.readFile ../config.toml))).out;
  schema = pkgs.writeText "graphiv-config-schema.json" (
    builtins.toJSON {
      inherit order;
      sections = describe [ ] options.services.graphiv.settings;
    }
  );
  override = "${toString cfg.stateDir}/config-override.toml";
  configEnv = {
    GRAPHIV_CONFIG = "${file}";
    GRAPHIV_CONFIG_OVERRIDE = override;
    GRAPHIV_CONFIG_SCHEMA = "${schema}";
  };

  # where systemd puts the manager's Hugging Face token, readable by that unit alone
  credential = "/run/credentials/graphiv-manager.service/hf-token";

  # the setgid directories a human in the group shares with the daemons
  shared = lib.unique [
    (toString cfg.stateDir)
    (toString cfg.dataDir)
    s.serve.CACHE_PATH
    s.serve.RUNS_PATH
    s.manager.STATE_PATH
    s.export.EXPORT_PATH
  ];

  mounts = lib.unique (
    shared
    ++ [
      s.dataset.HTML_PATH
      s.dataset.PDF_PATH
      s.dataset.GRAPH_PATH
    ]
    ++ lib.optional pg.enable (toString pg.dataDir)
  );

  store = lib.optional pg.enable "graphiv-pg.service";
  venv = toString cfg.venvDir;

  hardened = {
    User = cfg.user;
    Group = cfg.group;
    UMask = "0002";
    Restart = "on-failure";
    RestartSec = 10;
    NoNewPrivileges = true;
    PrivateTmp = true;
    ProtectSystem = "strict";
    ProtectKernelTunables = true;
    ProtectControlGroups = true;
    RestrictSUIDSGID = true;
  };

  listener = role: {
    description = "GraphIV ${role}";
    wantedBy = [ "multi-user.target" ];
    wants = [ "network-online.target" ];
    after = [
      "network-online.target"
      "graphiv-venv.service"
    ]
    ++ store
    ++ cfg.endpointUnits;
    requires = [ "graphiv-venv.service" ] ++ lib.optionals (role == "api") store;
    unitConfig.RequiresMountsFor = mounts;
    environment = configEnv // {
      GRAPHIV_VENV = venv;
    };
    serviceConfig = hardened // {
      ExecStart = "${cfg.package}/bin/graphiv-serve --http --role ${role}";
      # the api loads a 400MB graph export before it answers anything
      TimeoutStartSec = "5min";
      ReadWritePaths = lib.unique [
        s.serve.CACHE_PATH
        s.serve.RUNS_PATH
        s.dataset.HTML_PATH
        s.dataset.PDF_PATH
      ];
      ReadOnlyPaths = [ s.dataset.GRAPH_PATH ];
    };
  };

  pgValue = v: if lib.isBool v then (if v then "on" else "off") else toString v;
  pgFlags = lib.concatStringsSep " " (
    lib.mapAttrsToList (k: v: "-c ${lib.escapeShellArg "${k}=${pgValue v}"}") pg.settings
  );
  pgBin = "${pg.package}/bin";
  pgAt = "-h ${pg.socketDir} -p ${toString pg.port}";
in
{
  options.services.graphiv = {
    enable = lib.mkEnableOption "GraphIV, a grounded literature review endpoint over arXiv";

    package = lib.mkPackageOption pkgs "graphiv" { };

    user = opt types.str "graphiv" ''
      The unix user every unit runs as, and the owner of every directory below.
    '';

    group = opt types.str "graphiv" ''
      The group owning those directories, which a human member can also write.
    '';

    stateDir = opt types.path "/var/lib/graphiv" ''
      The root the cache, the runs, the manager's state and the eval output default under.
    '';

    dataDir = derived types.path "${toString cfg.stateDir}/data" ''"''${stateDir}/data"'' ''
      The cold data: the snapshot, the corpus, the graph export, the models and the cluster.
    '';

    venvDir = derived types.path "${toString cfg.stateDir}/venv" ''"''${stateDir}/venv"'' ''
      The venv graphiv-venv fills from uv.lock on the package's own python, which every unit runs.
    '';

    endpointUnits = opt (types.listOf types.str) [ ] ''
      Units serving the endpoints in settings.endpoints, which the api and the manager start after.
    '';

    postgres = {
      enable = opt types.bool true ''
        Run the cluster as graphiv-pg, one postmaster of its own over a unix socket.
        Turn it off to point settings.database.PG_DEST at a cluster run elsewhere.
      '';

      package = derived types.package (pkgs.postgresql_17.withPackages (p: [
        p.pgvector
      ])) "pkgs.postgresql_17.withPackages (p: [ p.pgvector ])" "The server, which needs pgvector.";

      dataDir = derived types.path "${toString cfg.dataDir}/pg" ''"''${dataDir}/pg"'' ''
        The cluster, created by initdb on first start when it holds no PG_VERSION.
      '';

      socketDir = opt types.str "/run/graphiv" ''
        Where the socket goes, a directory under /run that the unit creates.
      '';

      port = opt types.port 5433 ''
        The port, which names the socket file even though nothing listens on TCP.
      '';

      database = opt types.str "arxivkg" ''
        The database, created with the vector extension when the cluster has none by this name.
      '';

      settings =
        opt
          (types.attrsOf (
            types.oneOf [
              types.bool
              types.int
              types.float
              types.str
            ]
          ))
          {
            listen_addresses = "";
            default_toast_compression = "lz4";
            shared_buffers = "4GB";
            work_mem = "256MB";
            maintenance_work_mem = "2GB";
            max_parallel_maintenance_workers = 4;
          }
          "Passed as -c flags, which take precedence over the cluster's own postgresql.conf.";
    };

    api = {
      enable = opt types.bool true "Serve the six tools over streamable-http.";

      readOnly = opt types.bool false ''
        Register every tool but save_report, the only one that writes a file.
      '';

      restartOnExport = opt types.bool true ''
        Restart the api when the graph export is rewritten, since it loads the file once at start.
      '';
    };

    frontend.enable = opt types.bool true ''
      Serve the dashboard from a second process carrying no tools.
      It opens neither the store nor the graph, so it cannot reach the corpus.
    '';

    manager = {
      enable = opt types.bool true ''
        Serve the page that fetches the snapshot and runs every preprocess stage, one at a time.
        It writes the store and the corpus and loads the GPU, so keep it off a public vhost.
      '';

      databaseUser = opt (types.nullOr types.str) null ''
        The role the stages connect as, passed as PGUSER, when it is not the unit's own user.
        The stages insert, truncate and copy, so it has to own the four tables.
      '';

      path = opt (types.listOf types.package) [
        pkgs.aria2
        pkgs.curl
      ] ''
        Programs the stages run: aria2c for the PDFs, and nvidia-smi when the GPU should show.
        postgres.package joins them when postgres is enabled, for pg_dump, and otherwise name one.
      '';

      huggingface.tokenFile = opt (types.nullOr types.str) null ''
        A file holding the Hugging Face token the publish stage uploads with, such as a sops secret.
        systemd reads it as a credential of the manager's unit alone, so no other unit and no cache sees it.
        Name it as a string rather than a path literal, which would copy the token into the store.
      '';
    };

    settings = mkOption {
      default = { };
      description = "config.toml, one attribute per key, written into the store and read by every unit.";
      type = types.submodule {
        options = {
          database = section "Database" {
            PG_DEST =
              derived types.str "host=${pg.socketDir} port=${toString pg.port} dbname=${pg.database}"
                ''"host=''${postgres.socketDir} port=''${postgres.port} dbname=''${postgres.database}"''
                "Postgres connection string (libpq)";
          };

          source = section "Source URLs" {
            SOURCE_URL =
              opt types.str "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"
                "Snapshot download URL (Kaggle)";
            SOURCE_PATH =
              derived types.str "${toString cfg.dataDir}/arxiv.json" ''"''${dataDir}/arxiv.json"''
                "Snapshot in use";
            PDF_BASE_URL =
              opt types.str "https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf"
                "Bulk PDF mirror";
            PDF_LEGACY_BASE_URL = opt types.str "https://storage.googleapis.com/arxiv-dataset/arxiv" "PDF mirror for old slash ids";
            HTML_BASE_URL = opt types.str "https://arxiv.org/html" "Primary HTML source";
            HTML_FALLBACK_URL = opt types.str "https://ar5iv.labs.arxiv.org/html" "Fallback HTML source";
          };

          download = section "Download Settings" {
            PDF_BATCH = opt types.int 500 "Papers per aria2c call";
            PDF_N_PARALLEL = opt types.int 8 "Connections per aria2c call";
            HTML_RATE = opt types.float 8.0 "Max HTML requests per second";
            HTML_BATCH = opt types.int 200 "Papers per HTML batch";
            HTML_N_PARALLEL = opt types.int 16 "HTML download workers";
          };

          dataset = section "Storage Paths" {
            PDF_PATH = derived types.str "${toString cfg.dataDir}/arxiv/pdfs" ''"''${dataDir}/arxiv/pdfs"'' "PDF full text (real path, no symlink)";
            HTML_PATH = derived types.str "${toString cfg.dataDir}/arxiv/html" ''"''${dataDir}/arxiv/html"'' "HTML full text (real path, no symlink)";
            GRAPH_PATH =
              derived types.str "${toString cfg.dataDir}/citation_graph_export.npy"
                ''"''${dataDir}/citation_graph_export.npy"''
                "Citation graph export";
          };

          serve = section "MCP Server and Library" {
            HOST = opt types.str "127.0.0.1" "Bind address for every listener";
            API_PORT = opt types.port 6767 "MCP tool server port";
            FRONTEND_PORT = opt types.port 6969 "Research library port";
            ALLOWED_HOSTS = opt (types.listOf types.str) [ ] "Extra Host headers accepted (proxy)";
            RECIPE_PATH =
              derived types.str "${cfg.package}/share/graphiv/retrieval_recipes"
                ''"''${package}/share/graphiv/retrieval_recipes"''
                "Tuned retrieval recipes";
            RUNS_PATH = derived types.str "${toString cfg.stateDir}/runs" ''"''${stateDir}/runs"'' "Saved report runs";
            CACHE_PATH = derived types.str "${toString cfg.stateDir}/cache" ''"''${stateDir}/cache"'' "Model reply cache (safe to delete)";
            FETCH = opt types.bool true "Fetch missing papers from arXiv";
            PDFTOTEXT =
              derived types.str (lib.getExe' pkgs.poppler-utils "pdftotext")
                ''"''${poppler-utils}/bin/pdftotext"''
                "pdftotext binary (no-HTML fallback)";
          };

          retrieve = section "Retrieval" {
            COSINE_LIMIT = opt types.int 5000 "Papers from the cosine retriever";
            EXPANSION_LIMIT = opt types.int 20000 "Max papers per expander";
            DEPTH = opt types.int 4 "Hops the decay terms reach";
            DECAY = opt types.float 0.5 "Decay weight kept per hop";
          };

          miner = section "LeyLine Miner" {
            FOCUS = opt types.float 4.0 "Closeness exponent for magnets";
            WALK_PREFIX = opt types.int 30 "Ranking head the walks use";
            CONNECTORS = opt types.int 40 "Max bridging papers";
            DECAY = opt types.float 0.5 "Reach weight kept per hop";
            KNOCKDOWN = opt types.float 0.4 "Weight left once a line takes a paper";
            SEEDS = opt types.int 10 "Papers the walks start from";
            WALKS = opt types.int 50 "Walks per seed";
            BRANCH = opt types.int 1 "Papers per walk step";
            ARGUES = opt (types.listOf types.str) [
              "extends"
              "critiques"
              "compares"
            ] "Intents that count as arguing";
            SUCCESSOR_DAYS = opt types.int 240 "Successor window before the cap (days)";
            SUCCESSOR_FLOOR = opt types.float 0.6 "Min intent score for a successor";
            weights = section "LeyLine Intent Weights" {
              extends = opt types.float 1.0 "Weight of extends citations";
              critiques = opt types.float 1.0 "Weight of critiques citations";
              compares = opt types.float 0.8 "Weight of compares citations";
              uses = opt types.float 0.4 "Weight of uses citations";
              motivates = opt types.float 0.15 "Weight of motivates citations";
              background = opt types.float 0.0 "Weight of background citations";
            };
          };

          manager = section "Manager" {
            PORT = opt types.port 6868 "Manager port";
            STATE_PATH = derived types.str "${toString cfg.stateDir}/manager" ''"''${stateDir}/manager"'' "Job logs, charts and measured rates";
            HF_TOKEN_PATH =
              derived types.str (lib.optionalString (cfg.manager.huggingface.tokenFile != null) credential)
                ''"${credential}" when manager.huggingface.tokenFile is set, and "" otherwise''
                "Hugging Face token file";
          };

          export = section "Export and Publishing" {
            EXPORT_PATH = derived types.str "${toString cfg.dataDir}/export" ''"''${dataDir}/export"'' "Store dumps and Parquet releases";
            DATA_REPO = opt types.str "" "Data release HF repo (empty skips)";
            CITATION_REPO = opt types.str "" "Intent classifier HF repo (empty skips)";
            SURVEY_REPO = opt types.str "" "Review classifier HF repo (empty skips)";
            BENCHMARK_REPO = opt types.str "" "Benchmarks HF repo (empty skips)";
            QUERY_CACHE = derived types.str "${toString cfg.stateDir}/cache/queries" ''"''${stateDir}/cache/queries"''
              "Cached benchmark questions";
          };

          endpoints = section "Model Endpoints" {
            EMBEDDING_ENDPOINT = opt types.str "http://127.0.0.1:11434/embedding/nomic-embed-text-v2-moe" "llama-server URL (needs /tokenize)";
            EMBEDDING_API_KEY = opt types.str "" "Embedding API key (set in the manager)";
            EMBEDDING_NAME = opt types.str "nomic-embed-text-v2-moe" "Embedding model (must match the corpus)";
            EMBEDDING_PARALLEL = opt types.int 1 "Concurrent embedding requests";
            EMBEDDING_BATCH = opt types.int 256 "Documents per embedding request";
            EMBEDDING_MAX_SEQ = opt types.int 512 "Max tokens per document";
            EMBEDDING_DOCUMENT_PREFIX = opt types.str "search_document: " "Document prefix";
            EMBEDDING_QUERY_PREFIX = opt types.str "search_query: " "Question prefix";
            LLM_ENDPOINT = opt types.str "http://127.0.0.1:11434/v1" "Chat model URL (OpenAI compatible)";
            LLM_API_KEY = opt types.str "" "Chat API key (set in the manager)";
            LLM_NAME = mkOption {
              type = types.str;
              example = "Wordslop-Qwen3.6-27B";
              description = "Chat model";
            };
          };

          survey = section "Review Classifier" {
            MODEL_PATH =
              derived types.str "${toString cfg.dataDir}/survey_detection" ''"''${dataDir}/survey_detection"''
                "Review classifier checkpoint";
            BATCH = opt types.int 64 "Papers per inference batch";
            CHUNK = opt types.int 50000 "Papers per database commit";
          };

          umap = section "UMAP Projection" {
            UMAP_MODEL_PATH =
              derived types.str "${toString cfg.dataDir}/umap_50.joblib" ''"''${dataDir}/umap_50.joblib"''
                "Fitted UMAP model (pickle)";
            UMAP_SAMPLE = opt types.int 400000 "Papers sampled for a refit";
            UMAP_SEED = opt types.int 69420 "Seed (forces a single-core refit)";
            UMAP_DIM = opt types.int 50 "Output dimensions";
            UMAP_BATCH = opt types.int 50000 "Papers per transform call";
            UMAP_WORKERS = opt types.int 4 "Transform processes";
          };

          topics = section "Topic Tree" {
            TREE_METHODS = opt (types.listOf types.str) [
              "kmeans"
              "kmeans"
              "kmeans"
            ] "Clustering method per level";
            TREE_K = opt types.int 16 "Children per node";
            TREE_MIN_CLUSTER_SIZE = opt types.int 60 "Min papers per leaf";
            TREE_N_PARALLEL = opt types.int 8 "Clustering workers";
            TREE_BATCH = opt types.int 50000 "Rows per database write";
            TREE_SEED = opt types.int 69420 "Clustering seed";
            N_TOPIC_LABEL = opt types.int 10 "Papers shown per leaf when naming";
            TOPIC_BATCH = opt types.int 8 "Concurrent naming calls";
          };

          citation_labeler = section "Citation Labeler" {
            LABELER_PATH =
              derived types.str "${toString cfg.dataDir}/citation_labeler" ''"''${dataDir}/citation_labeler"''
                "Student checkpoint and ground truth";
            LABELER_MODEL = opt types.str "distilbert/distilbert-base-uncased" "Student base model";
            LABELER_BATCH = opt types.int 48 "Sentences per inference batch";
            LABELER_MAX_SEQ = opt types.int 256 "Max tokens per sentence";
            LABELER_APE_MODEL = derived types.str s.endpoints.LLM_NAME "settings.endpoints.LLM_NAME" "Teacher model for prompt evolution";
            LABELER_APE_GENERATIONS = opt types.int 6 "Prompt evolution rounds";
            LABELER_APE_PARALLEL = opt types.int 4 "Concurrent teacher calls";
            LABELER_TEACHER_SAMPLE = opt types.int 10000 "Citations the teacher labels";
            LABELER_SEED = opt types.int 0 "Sampling and split seed";
            LABELER_SPLITS = opt (types.listOf types.float) [ 0.8 0.1 0.1 ] "Train, validation, test fractions";
            LABELER_EPOCHS = opt types.int 12 "Student training epochs";
            LABELER_TRAIN_BATCH = opt types.int 64 "Sentences per training batch";
            LABELER_LR = opt types.float 2.0e-5 "Student learning rate";
          };

          surveymatch = section "Review Case Selection" {
            SURVEY_THRESHOLD = opt types.float 0.5 "Min review classifier score";
            MUTUAL_COSINE_THRESHOLD = opt types.float 0.65 "Min cosine between a case's reviews";
            SURVEYMATCH_LEVEL = opt types.int 2 "Topic tree level cases group by";
          };

          benchmarks = section "Benchmark Release" {
            CUTOFF = opt types.str "2026-10-22" "Last mined review date (release version)";
            SURVEYMATCH_FLOOR = opt types.int 100 "Min refs per SurveyMatch review";
            FRONTIERMATCH_FLOOR = opt types.int 25 "Min refs per FrontierMatch review";
            CONSENSUSMATCH_FLOOR = opt types.int 25 "Min refs per ConsensusMatch review";
            SURVEY_THRESHOLD = opt types.float 0.95 "Min review classifier score";
            GAP_DAYS = opt types.int 190 "Max gap between a case's reviews (days)";
            FRONTIER_COSINE = opt types.float 0.8 "Min cosine between FrontierMatch reviews";
            FRONTIER_MIN_KEY = opt types.int 10 "Min FrontierMatch answer key size";
            FRONTIER_WINDOW_DAYS = opt types.int 730 "Max age of a frontier answer (days)";
            CONSENSUS_COSINE = opt types.float 0.75 "Min cosine among ConsensusMatch reviews";
            CONSENSUS_MIN_KEY = opt types.int 5 "Min ConsensusMatch answer key size";
          };
        };
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.api.enable || cfg.frontend.enable || cfg.manager.enable;
        message = "services.graphiv enables no unit";
      }
      {
        assertion =
          cfg.manager.huggingface.tokenFile == null
          || !lib.hasPrefix builtins.storeDir cfg.manager.huggingface.tokenFile;
        message = "services.graphiv.manager.huggingface.tokenFile is in the store, which every user can read";
      }
      {
        assertion = lib.allUnique [
          s.serve.API_PORT
          s.serve.FRONTEND_PORT
          s.manager.PORT
        ];
        message = "services.graphiv.settings needs API_PORT, FRONTEND_PORT and manager.PORT to differ";
      }
      {
        assertion = s.endpoints.LLM_API_KEY == "" && s.endpoints.EMBEDDING_API_KEY == "";
        message = "services.graphiv.settings.endpoints API keys would land in the world-readable store, so set them in the manager";
      }
      {
        assertion = !pg.enable || lib.hasPrefix "/run/" pg.socketDir;
        message = "services.graphiv.postgres.socketDir has to sit under /run, which the unit creates";
      }
    ];

    users.users = lib.mkIf (cfg.user == "graphiv") {
      graphiv = {
        isSystemUser = lib.mkDefault true;
        group = lib.mkDefault cfg.group;
        home = lib.mkDefault (toString cfg.stateDir);
        description = lib.mkDefault "GraphIV";
      };
    };

    users.groups = lib.mkIf (cfg.group == "graphiv") { graphiv = { }; };

    # setgid so a human in the group can read and delete what a daemon wrote
    # A d rule touches only the directory it names, so a symlinked corpus is left alone
    # the default sits beside the override as a link into the store, and the override holds the API keys
    systemd.tmpfiles.rules =
      map (d: "d ${d} 2775 ${cfg.user} ${cfg.group} - -") shared
      ++ lib.optional pg.enable "d ${toString pg.dataDir} 0700 ${cfg.user} ${cfg.group} - -"
      ++ [
        "L+ ${toString cfg.stateDir}/config.toml - - - - ${file}"
        "f ${override} 0600 ${cfg.user} ${cfg.group} - -"
      ];

    systemd.services.graphiv-pg = lib.mkIf pg.enable {
      description = "GraphIV postgres";
      wantedBy = [ "multi-user.target" ];
      unitConfig.RequiresMountsFor = [ (toString pg.dataDir) ];
      preStart = ''
        if [ ! -e ${toString pg.dataDir}/PG_VERSION ]; then
          ${pgBin}/initdb -D ${toString pg.dataDir} -U ${cfg.user} --encoding=UTF8 --no-locale --auth-local=peer
        fi
      '';
      # a failed probe leaves the cluster up, since this only fills in a missing database
      postStart = ''
        found=$(${pgBin}/psql ${pgAt} -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '${pg.database}'") || exit 0
        if [ -z "$found" ]; then
          ${pgBin}/createdb ${pgAt} ${pg.database}
          ${pgBin}/psql ${pgAt} -d ${pg.database} -c "CREATE EXTENSION IF NOT EXISTS vector"
        fi
      '';
      serviceConfig = {
        Type = "notify";
        ExecStart = "${pgBin}/postgres -D ${toString pg.dataDir} -k ${pg.socketDir} -p ${toString pg.port} ${pgFlags}";
        User = cfg.user;
        Group = cfg.group;
        RuntimeDirectory = lib.removePrefix "/run/" pg.socketDir;
        RuntimeDirectoryMode = "0755";
        Restart = "on-failure";
        RestartSec = 10;
        TimeoutStartSec = "5min";
        TimeoutStopSec = "10min";
        KillSignal = "SIGINT";
        KillMode = "mixed";
        # logind clears the POSIX shm of a uid >= 1000 on logout, which kills a backend using it
        # A system user is already exempt, and this says so out loud
        RemoveIPC = false;
        OOMScoreAdjust = -900;
      };
    };

    # uv installs every Python dependency here, and a new lock or a new python re-runs it on switch
    systemd.services.graphiv-venv = {
      description = "GraphIV venv, synced from uv.lock";
      wants = [ "network-online.target" ];
      after = [ "network-online.target" ];
      unitConfig.RequiresMountsFor = [
        venv
        s.serve.CACHE_PATH
      ];
      environment = {
        GRAPHIV_VENV = venv;
        UV_CACHE_DIR = "${s.serve.CACHE_PATH}/uv";
      };
      serviceConfig = hardened // {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = "${cfg.package}/bin/graphiv-sync";
        # the first sync downloads torch and its CUDA wheels
        TimeoutStartSec = "2h";
        ReadWritePaths = lib.unique [
          (dirOf venv)
          s.serve.CACHE_PATH
        ];
      };
    };

    systemd.services.graphiv-api = lib.mkIf cfg.api.enable (
      lib.recursiveUpdate (listener "api") {
        environment.GRAPHIV_READONLY = lib.mkIf cfg.api.readOnly "1";
      }
    );

    systemd.services.graphiv-frontend = lib.mkIf cfg.frontend.enable (listener "frontend");

    systemd.services.graphiv-manager = lib.mkIf cfg.manager.enable {
      description = "GraphIV manager";
      wantedBy = [ "multi-user.target" ];
      wants = [ "network-online.target" ];
      after = [
        "network-online.target"
        "graphiv-venv.service"
      ]
      ++ store
      ++ cfg.endpointUnits;
      requires = [ "graphiv-venv.service" ] ++ store;
      unitConfig.RequiresMountsFor = mounts;
      # the dump stage runs pg_dump, which cannot be older than the cluster
      path = cfg.manager.path ++ lib.optional pg.enable pg.package;
      environment = configEnv // {
        GRAPHIV_VENV = venv;
        HF_HOME = "${s.serve.CACHE_PATH}/huggingface";
        # umap and pynndescent compile with numba, whose cache cannot sit beside the store's source
        NUMBA_CACHE_DIR = "${s.serve.CACHE_PATH}/numba";
      }
      // lib.optionalAttrs (cfg.manager.databaseUser != null) { PGUSER = cfg.manager.databaseUser; };
      serviceConfig = hardened // {
        ExecStart = "${cfg.package}/bin/graphiv-manager --config_path ${file}";
        LoadCredential = lib.optional (
          cfg.manager.huggingface.tokenFile != null
        ) "hf-token:${cfg.manager.huggingface.tokenFile}";
        # a stop sends the running stage SIGINT and waits for it to roll back, up to ten minutes
        KillMode = "mixed";
        TimeoutStopSec = "15min";
        ReadWritePaths = lib.unique (
          shared
          ++ [
            s.dataset.HTML_PATH
            s.dataset.PDF_PATH
            "-${dirOf s.source.SOURCE_PATH}"
            "-${dirOf s.dataset.GRAPH_PATH}"
            "-${dirOf s.umap.UMAP_MODEL_PATH}"
          ]
        );
      };
    };

    systemd.paths.graphiv-config = lib.mkIf (cfg.api.enable || cfg.frontend.enable) {
      description = "GraphIV config override";
      wantedBy = [ "paths.target" ];
      unitConfig.RequiresMountsFor = [ (toString cfg.stateDir) ];
      pathConfig.PathChanged = override;
    };

    systemd.services.graphiv-config = lib.mkIf (cfg.api.enable || cfg.frontend.enable) {
      description = "Restart graphiv-api and graphiv-frontend onto a new config override";
      serviceConfig.Type = "oneshot";
      script = "systemctl try-restart graphiv-api.service graphiv-frontend.service";
    };

    systemd.paths.graphiv-graph = lib.mkIf (cfg.api.enable && cfg.api.restartOnExport) {
      description = "GraphIV graph export";
      wantedBy = [ "paths.target" ];
      # a watch set before the mount would sit on the empty mountpoint
      unitConfig.RequiresMountsFor = [ s.dataset.GRAPH_PATH ];
      pathConfig.PathChanged = s.dataset.GRAPH_PATH;
    };

    systemd.services.graphiv-graph = lib.mkIf (cfg.api.enable && cfg.api.restartOnExport) {
      description = "Restart graphiv-api onto a new graph export";
      serviceConfig.Type = "oneshot";
      script = "systemctl try-restart graphiv-api.service";
    };
  };
}
