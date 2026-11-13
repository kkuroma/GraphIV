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
          database = section "The store." {
            PG_DEST =
              derived types.str "host=${pg.socketDir} port=${toString pg.port} dbname=${pg.database}"
                ''"host=''${postgres.socketDir} port=''${postgres.port} dbname=''${postgres.database}"''
                "The libpq connection string every unit and stage opens.";
          };

          source = section "Where the snapshot and the full text come from." {
            SOURCE_URL =
              opt types.str "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"
                "The snapshot archive, which Kaggle serves anonymously as a redirect.";
            SOURCE_PATH =
              derived types.str "${toString cfg.dataDir}/arxiv.json" ''"''${dataDir}/arxiv.json"''
                "The snapshot in use, with a fetched one staged beside it as arxiv-new.json.";
            PDF_BASE_URL =
              opt types.str "https://storage.googleapis.com/arxiv-dataset/arxiv/arxiv/pdf"
                "The bulk PDF mirror.";
            PDF_LEGACY_BASE_URL = opt types.str "https://storage.googleapis.com/arxiv-dataset/arxiv" ''
              The mirror's layout for the old slash ids.
            '';
            HTML_BASE_URL = opt types.str "https://arxiv.org/html" "The first HTML source.";
            HTML_FALLBACK_URL = opt types.str "https://ar5iv.labs.arxiv.org/html" ''
              The HTML source tried when the first has no render.
            '';
          };

          download = section "How hard the corpus stages pull." {
            PDF_BATCH = opt types.int 500 "Papers per aria2c invocation.";
            PDF_N_PARALLEL = opt types.int 8 "Connections per aria2c invocation.";
            HTML_RATE = opt types.float 8.0 ''
              Requests a second over every worker, where a miss on the first source costs two.
            '';
            HTML_BATCH = opt types.int 200 "Papers per HTML batch.";
            HTML_N_PARALLEL = opt types.int 16 "HTML workers.";
          };

          dataset = section "The corpus and the graph export." {
            PDF_PATH = derived types.str "${toString cfg.dataDir}/arxiv/pdfs" ''"''${dataDir}/arxiv/pdfs"'' ''
              A tree of YYMM/<id>v<n>.pdf.
              Name the real path rather than a symlink, so the units order on the right mount.
            '';
            HTML_PATH = derived types.str "${toString cfg.dataDir}/arxiv/html" ''"''${dataDir}/arxiv/html"'' ''
              A tree of YYMM/<id>.html, on the same terms, which a fetch also writes into.
            '';
            GRAPH_PATH =
              derived types.str "${toString cfg.dataDir}/citation_graph_export.npy"
                ''"''${dataDir}/citation_graph_export.npy"''
                "The citation graph export, which the api loads once and the last stage rewrites.";
          };

          serve = section "The two listeners." {
            HOST = opt types.str "127.0.0.1" "The interface every listener binds, the manager's too.";
            API_PORT = opt types.port 6767 "The tools.";
            FRONTEND_PORT = opt types.port 6969 "The dashboard.";
            ALLOWED_HOSTS = opt (types.listOf types.str) [ ] ''
              Host headers the transport answers to beyond its own address, which a proxy forwards.
            '';
            RECIPE_PATH =
              derived types.str "${cfg.package}/share/graphiv/retrieval_recipes"
                ''"''${package}/share/graphiv/retrieval_recipes"''
                "The tuned recipes, which ship with the package.";
            RUNS_PATH = derived types.str "${toString cfg.stateDir}/runs" ''"''${stateDir}/runs"'' ''
              One directory per report written, which the dashboard lists.
            '';
            CACHE_PATH = derived types.str "${toString cfg.stateDir}/cache" ''"''${stateDir}/cache"'' ''
              Model replies keyed by their prompt, safe to delete at any time.
            '';
            FETCH = opt types.bool true ''
              Whether a paper the corpus never had may be fetched from arXiv into the corpus.
            '';
            PDFTOTEXT =
              derived types.str (lib.getExe' pkgs.poppler-utils "pdftotext")
                ''"''${poppler-utils}/bin/pdftotext"''
                "The extractor read_paper falls back to for a paper with no usable HTML.";
          };

          retrieve = section "The retrieval pipeline every recipe runs." {
            COSINE_LIMIT = opt types.int 5000 "Papers the cosine retriever returns.";
            EXPANSION_LIMIT = opt types.int 20000 "Most papers each expander appends.";
            DEPTH = opt types.int 4 "Hops the decay terms reach.";
            DECAY = opt types.float 0.5 "Weight kept per hop in the decay terms.";
          };

          miner = section "The miner that turns a ranking into lines." {
            FOCUS = opt types.float 4.0 "Exponent on closeness when scoring an argued-with paper.";
            WALK_PREFIX = opt types.int 30 "Head of the ranking the walks run on.";
            CONNECTORS = opt types.int 40 "Most bridging papers kept.";
            DECAY = opt types.float 0.5 "Weight kept per hop of a paper's reach.";
            KNOCKDOWN = opt types.float 0.4 "Weight left on a paper once a line takes it.";
            SEEDS = opt types.int 10 "Papers the walks start from.";
            WALKS = opt types.int 50 "Walks per seed.";
            BRANCH = opt types.int 1 "Papers a walk steps to at a time.";
            ARGUES = opt (types.listOf types.str) [
              "extends"
              "critiques"
              "compares"
            ] "Intents that count as arguing with a paper.";
            SUCCESSOR_DAYS = opt types.int 240 "Days before the cap a successor may date from.";
            SUCCESSOR_FLOOR = opt types.float 0.6 "Lowest intent score a successor's citation needs.";
            weights = section "Weight of each citation intent in the miner." {
              extends = opt types.float 1.0 "Extends.";
              critiques = opt types.float 1.0 "Critiques.";
              compares = opt types.float 0.8 "Compares.";
              uses = opt types.float 0.4 "Uses.";
              motivates = opt types.float 0.15 "Motivates.";
              background = opt types.float 0.0 "Background.";
            };
          };

          manager = section "The manager." {
            PORT = opt types.port 6868 "The page.";
            STATE_PATH = derived types.str "${toString cfg.stateDir}/manager" ''"''${stateDir}/manager"'' ''
              Job logs, the cached charts and the rates it measured.
            '';
            HF_TOKEN_PATH =
              derived types.str (lib.optionalString (cfg.manager.huggingface.tokenFile != null) credential)
                ''"${credential}" when manager.huggingface.tokenFile is set, and "" otherwise''
                "Where the manager reads the Hugging Face token, empty when there is none.";
          };

          export = section "What leaves the store." {
            EXPORT_PATH = derived types.str "${toString cfg.dataDir}/export" ''"''${dataDir}/export"'' ''
              Where the manager writes graphiv-<date>.dump, the full backup, and graphiv-<date>/, the Parquet release, and graphiv-benchmarks-<cutoff>/.
            '';
            DATA_REPO = opt types.str "" "The Hugging Face dataset repo the publish stage uploads the release to, empty to skip it.";
            CITATION_REPO = opt types.str "" "The model repo for the citation intent classifier, empty to skip it.";
            SURVEY_REPO = opt types.str "" "The model repo for the survey classifier, empty to skip it.";
            BENCHMARK_REPO = opt types.str "" "The dataset repo for the benchmarks, empty to skip it.";
            QUERY_CACHE = derived types.str "${toString cfg.stateDir}/cache/queries" ''"''${stateDir}/cache/queries"''
              "Each benchmark question the model wrote, cached by its input so a re-export keeps it.";
          };

          endpoints = section "The models, all of them served elsewhere." {
            EMBEDDING_ENDPOINT = opt types.str "http://127.0.0.1:11434/embedding/nomic-embed-text-v2-moe" ''
              A llama-server hosting the embedding model, which must expose /tokenize, or a
              llama-router's path for one embedding model, which loads it on the first call.
            '';
            EMBEDDING_API_KEY = opt types.str "" ''
              The embedding endpoint's key, which stays empty here and is set in the manager.
            '';
            EMBEDDING_NAME = opt types.str "nomic-embed-text-v2-moe" ''
              The embedding model, which has to be the one the corpus was built with.
            '';
            EMBEDDING_PARALLEL = opt types.int 1 "Embedding requests in flight.";
            EMBEDDING_BATCH = opt types.int 256 "Documents per embedding request.";
            EMBEDDING_MAX_SEQ = opt types.int 512 "Tokens a document is truncated to, the two special tokens included.";
            EMBEDDING_DOCUMENT_PREFIX = opt types.str "search_document: " "What a stored document is prefixed with.";
            EMBEDDING_QUERY_PREFIX = opt types.str "search_query: " "What a question is prefixed with.";
            LLM_ENDPOINT = opt types.str "http://127.0.0.1:11434/v1" "Any OpenAI compatible host.";
            LLM_API_KEY = opt types.str "" ''
              The chat endpoint's key, which stays empty here and is set in the manager.
            '';
            LLM_NAME = mkOption {
              type = types.str;
              example = "Wordslop-Qwen3.6-27B";
              description = "The model every call goes to, a 26B instruct class being the reference.";
            };
          };

          survey = section "The literature review scorer." {
            MODEL_PATH =
              derived types.str "${toString cfg.dataDir}/survey_detection" ''"''${dataDir}/survey_detection"''
                ''
                  The checkpoint and the recipe that trained it.
                '';
            BATCH = opt types.int 64 "Papers per forward pass.";
            CHUNK = opt types.int 50000 "Papers per commit.";
          };

          umap = section "The projection the topic tree clusters." {
            UMAP_MODEL_PATH =
              derived types.str "${toString cfg.dataDir}/umap_50.joblib" ''"''${dataDir}/umap_50.joblib"''
                ''
                  The fitted model, a pickle that loads only under the umap-learn it was fitted with.
                '';
            UMAP_SAMPLE = opt types.int 400000 "Papers a refit samples.";
            UMAP_SEED = opt types.int 69420 "The seed, which forces a refit onto one core.";
            UMAP_DIM = opt types.int 50 "Dimensions.";
            UMAP_BATCH = opt types.int 50000 "Papers per transform call.";
            UMAP_WORKERS = opt types.int 4 "Transform processes, each loading its own copy of the model.";
          };

          topics = section "The topic tree and its names." {
            TREE_METHODS = opt (types.listOf types.str) [
              "kmeans"
              "kmeans"
              "kmeans"
            ] "The clusterer per level.";
            TREE_K = opt types.int 16 "Children per node.";
            TREE_MIN_CLUSTER_SIZE = opt types.int 60 "The smallest leaf.";
            TREE_N_PARALLEL = opt types.int 8 "Clustering workers.";
            TREE_BATCH = opt types.int 50000 "Rows per write.";
            TREE_SEED = opt types.int 69420 "The seed.";
            N_TOPIC_LABEL = opt types.int 10 "Papers shown to the namer, nearest the leaf centroid.";
            TOPIC_BATCH = opt types.int 8 "Naming calls in flight.";
          };

          citation_labeler = section "The citation intent classifier and the job that trains it." {
            LABELER_PATH =
              derived types.str "${toString cfg.dataDir}/citation_labeler" ''"''${dataDir}/citation_labeler"''
                ''
                  The student checkpoint and the ground truth it is scored on.
                '';
            LABELER_MODEL = opt types.str "distilbert/distilbert-base-uncased" "The student's base.";
            LABELER_BATCH = opt types.int 48 "Sentences per forward pass.";
            LABELER_MAX_SEQ = opt types.int 256 "Tokens a sentence is truncated to.";
            LABELER_APE_MODEL = derived types.str s.endpoints.LLM_NAME "settings.endpoints.LLM_NAME" ''
              The teacher that evolves the prompt.
            '';
            LABELER_APE_GENERATIONS = opt types.int 6 "Prompt generations.";
            LABELER_APE_PARALLEL = opt types.int 4 "Teacher calls in flight.";
            LABELER_TEACHER_SAMPLE = opt types.int 10000 "Citations the teacher labels.";
            LABELER_SEED = opt types.int 0 "The seed.";
            LABELER_SPLITS = opt (types.listOf types.float) [ 0.8 0.1 0.1 ] "Train, validation and test.";
            LABELER_EPOCHS = opt types.int 12 "Epochs.";
            LABELER_TRAIN_BATCH = opt types.int 64 "Training batch.";
            LABELER_LR = opt types.float 2.0e-5 "Learning rate.";
          };

          surveymatch = section "Review case selection for the benchmarks." {
            SURVEY_THRESHOLD = opt types.float 0.5 "The review score a case needs.";
            MUTUAL_COSINE_THRESHOLD = opt types.float 0.65 "The cosine two reviews in one case share.";
            SURVEYMATCH_LEVEL = opt types.int 2 "The tree level cases group by.";
          };

          benchmarks = section "The published benchmarks, versioned by their cutoff." {
            CUTOFF = opt types.str "2026-10-22" "The last day a mined review may carry, which names the version.";
            SURVEYMATCH_FLOOR = opt types.int 100 "Resolved references a SurveyMatch review needs.";
            FRONTIERMATCH_FLOOR = opt types.int 25 "Resolved references a FrontierMatch review needs.";
            CONSENSUSMATCH_FLOOR = opt types.int 25 "Resolved references a ConsensusMatch review needs.";
            SURVEY_THRESHOLD = opt types.float 0.95 "The review score a case needs.";
            GAP_DAYS = opt types.int 190 "The widest gap between two reviews of one moment.";
            FRONTIER_COSINE = opt types.float 0.8 "The cosine two FrontierMatch reviews share.";
            FRONTIER_MIN_KEY = opt types.int 10 "The smallest FrontierMatch answer key.";
            FRONTIER_WINDOW_DAYS = opt types.int 730 "How early a frontier answer may be published.";
            CONSENSUS_COSINE = opt types.float 0.75 "The cosine three ConsensusMatch reviews share.";
            CONSENSUS_MIN_KEY = opt types.int 5 "The smallest ConsensusMatch answer key.";
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
