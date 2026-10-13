{
  description = "SystemNone: the Rust gateway and the Python training worker, as packages and container images";

  # Not nixos-26.05: its torch 2.11 and transformers 5.5 fail the worker's ONNX export.
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (
        pkgs:
        let
          inherit (pkgs) lib;
          python = pkgs.python3;
          onnxruntimeLibrary = "${lib.getLib pkgs.onnxruntime}/lib/libonnxruntime.so";

          gateway = pkgs.rustPlatform.buildRustPackage {
            pname = "system-none-gateway";
            version = "0.1.0";
            src = ./gateway;
            cargoLock.lockFile = ./gateway/Cargo.lock;
            nativeBuildInputs = [ pkgs.makeWrapper ];
            # The integration test starts a gateway, which loads ONNX Runtime and CA certificates.
            preCheck = ''
              export ORT_DYLIB_PATH=${onnxruntimeLibrary}
              export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            '';
            postInstall = ''
              wrapProgram $out/bin/system-none-gateway --set-default ORT_DYLIB_PATH ${onnxruntimeLibrary}
            '';
            meta.mainProgram = "system-none-gateway";
          };

          # The library, CLI and worker, with the train and s3 extras.
          system-none = python.pkgs.buildPythonApplication {
            pname = "system-none";
            version = "0.1.0";
            pyproject = true;
            src = lib.fileset.toSource {
              root = ./.;
              fileset = lib.fileset.unions [
                ./pyproject.toml
                ./README.md
                ./LICENSE
                ./src
              ];
            };
            build-system = with python.pkgs; [
              setuptools
              wheel
            ];
            dependencies = with python.pkgs; [
              numpy
              pyyaml
              httpx
              langid
              tokenizers
              onnxruntime
              torch
              transformers
              sentencepiece
              protobuf
              onnx
              onnxscript
              sentence-transformers
              boto3
            ];
            pythonImportsCheck = [ "system_none" ];
            meta.mainProgram = "system-none";
          };

          # CI publishes the images here (the Images workflow); docker-compose.yaml pulls them.
          registry = "ghcr.io/jaderdias";

          # What each container runs.
          containers = {
            gateway = {
              package = gateway;
              config = {
                Entrypoint = [ "/bin/system-none-gateway" ];
                Env = [
                  "SN_LISTEN=0.0.0.0:8080"
                  "SN_DB=/data/gateway.db"
                  "SN_STORE=file:///data/store"
                  "SN_CACHE=/data/models"
                ];
                ExposedPorts."8080/tcp" = { };
              };
            };
            worker = {
              package = system-none;
              config = {
                Entrypoint = [
                  "/bin/system-none"
                  "worker"
                ];
                Cmd = [
                  "--gateway"
                  "http://gateway:8080"
                ];
                Env = [ "HF_HOME=/data/huggingface" ];
              };
            };
          };

          # Both run as nobody, who owns only /data and /tmp. The gateway tells the worker where
          # its store is, so with a file:// store both containers mount the same volume at /data.
          configOf =
            c:
            c.config
            // {
              User = "65534:65534";
              WorkingDir = "/data";
              Volumes."/data" = { };
              Labels = {
                # Links the published package to the repository on GitHub.
                "org.opencontainers.image.source" = "https://github.com/JaderDias/quickstill";
                "org.opencontainers.image.licenses" = "Apache-2.0";
              };
              Env = [
                "SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"
                "HOME=/data"
              ]
              ++ c.config.Env;
            };

          # An image's files outside /nix/store: /bin from its package, /etc from cacert and fakeNss.
          rootOf =
            name: c:
            pkgs.buildEnv {
              name = "system-none-${name}-root";
              paths = [
                c.package
                pkgs.cacert
                pkgs.dockerTools.fakeNss
              ];
              pathsToLink = [
                "/bin"
                "/etc"
              ];
            };

          imageOf =
            name: c:
            pkgs.dockerTools.buildLayeredImage {
              name = "${registry}/system-none-${name}";
              tag = null; # the output hash, so `nix run .#compose` loads each build once
              contents = [ (rootOf name c) ];
              # /data belongs to nobody, so a fresh volume copies that ownership.
              fakeRootCommands = "mkdir -p data tmp && chmod 1777 tmp && chown 65534:65534 data";
              config = configOf c;
            };
          images = lib.mapAttrs imageOf containers;

          # "name:tag=archive" for each image, for the shell loops below.
          imageList = lib.concatStringsSep " " (
            lib.mapAttrsToList (_: i: "${i.imageName}:${i.imageTag}=${i}") images
          );

          # Builds and loads the images under the names docker-compose.yaml pulls, so Compose runs
          # these instead. The images are reproducible, so they match the published ones for the
          # same commit.
          compose = pkgs.writeShellApplication {
            name = "system-none-compose";
            runtimeInputs = [ pkgs.coreutils ]; # docker comes from the host
            text = ''
              for image in ${imageList}; do
                name="''${image%%=*}"
                docker image inspect "$name" >/dev/null 2>&1 || docker load < "''${image#*=}"
                docker tag "$name" "''${name%%:*}:latest"
              done
              if [ ! -e .env ]; then
                token() { od -An -N16 -tx1 /dev/urandom | tr -d ' \n'; }
                (umask 077; printf 'SN_ADMIN_TOKEN=%s\nSN_WORKER_TOKEN=%s\n' "$(token)" "$(token)" > .env)
                echo "wrote .env with new admin and worker tokens"
              fi
              echo "loaded the images; start with: docker compose up -d"
            '';
          };

          # usage: publish-images TAG...   pushes both images to the registry under each tag.
          # With REGISTRY_USER and REGISTRY_PASSWORD set it logs in first; the Images workflow
          # runs it on main.
          publish-images = pkgs.writeShellApplication {
            name = "system-none-publish-images";
            runtimeInputs = [ pkgs.skopeo ];
            text = ''
              if [ -n "''${REGISTRY_PASSWORD:-}" ]; then
                printf '%s' "$REGISTRY_PASSWORD" |
                  skopeo login --username "$REGISTRY_USER" --password-stdin ${lib.head (lib.splitString "/" registry)}
              fi
              for image in ${imageList}; do
                name="''${image%%:*}"
                for tag in "$@"; do
                  skopeo --insecure-policy copy --retry-times 3 "docker-archive:''${image#*=}" "docker://$name:$tag"
                done
              done
            '';
          };
        in
        {
          inherit
            gateway
            system-none
            compose
            publish-images
            ;
          gateway-image = images.gateway;
          worker-image = images.worker;
        }
      );
    };
}
