# SP5 task 5: the Hetzner burst tier's cluster (D3: OpenTofu + kube-hetzner k3s on Hetzner CX
# instances). This is the ENTIRE tofu contract: produce a k8s cluster + kubeconfig + node labels.
# Everything above that line -- every workload, every Secret, every Deployment -- is the SAME
# helmfile + charts the k3d tier already proves daily (D3: "one set of charts"); nothing in
# infra/charts/ or infra/environments/ is cloud specific, and nothing here reaches upward into either.
# See infra/README.md's "Cloud portability boundary" section for what that split buys.
#
# Sizing (HLD section 7.1, "Hetzner CX instances"; the exact split -- one CX23 control plane, two
# CX33 workers -- follows the SP5 planning digest's own sizing note): a CX23 control plane is
# deliberately small (it runs the k3s API server and etcd only, D1 already refused Postgres HA so
# this is not a quorum member of anything); CX33 workers carry the actual workload (CNPG, both TEI
# services, backend, web) and need the extra memory headroom the tei chart's own 18Gi/12Gi limits
# already assume live on real hardware, not Rosetta emulated arm64 (see infra/README.md's TEI
# section). `eu-central` (fsn1/nbg1/hel1) is the only region the CX server family is sold in.
provider "hcloud" {
  token = var.hcloud_token
}

module "kube_hetzner" {
  source  = "kube-hetzner/kube-hetzner/hcloud"
  version = "3.0.1" # pinned (D26: never a floating alias); see infra/README.md for the bump procedure

  providers = {
    hcloud = hcloud
  }

  hcloud_token     = var.hcloud_token
  ssh_public_key   = var.ssh_public_key
  ssh_private_key  = var.ssh_private_key
  cluster_name     = var.cluster_name
  base_domain      = var.base_domain
  network_region   = "eu-central"

  control_plane_nodepools = [
    {
      name        = "control-plane-fsn1"
      server_type = "cx23"
      location    = "fsn1"
      labels      = []
      taints      = []
      count       = 1
    },
  ]

  agent_nodepools = [
    {
      name        = "agent-fsn1"
      server_type = "cx33"
      location    = "fsn1"
      labels      = []
      taints      = []
      count       = 2
    },
  ]

  load_balancer_type     = "lb11"
  load_balancer_location = "fsn1"

  # D35: k3s ships Traefik as its default ingress controller; the module's own "traefik" mode
  # configures THAT instance (no second ingress controller), matching infra/charts/atlas-ingress's
  # own IngressRoute/Middleware, which target the same built in Traefik on both tiers.
  ingress_controller = "traefik"

  # Task 5: the wildcard cert story (infra/charts/atlas-cert, helmfile, DNS-01) needs the cert-manager
  # CRDs/operator already installed cluster side; this is the module's own default (true), set
  # explicitly here so the dependency is not an implicit accident, matching this repo's own
  # imagePullPolicy style discipline of asserting meaningful defaults rather than leaving them silent.
  enable_cert_manager = true

  # D40: "the kube-hetzner firewall limits the API server and SSH to the operator IP." The module's
  # own default is open to the world (0.0.0.0/0, ::/0); "myipv4" is kube-hetzner's own documented
  # placeholder, resolved at apply time (via icanhazip.com) to the machine actually running `tofu
  # apply`'s own public IPv4 /32 -- exactly "the operator IP," without this task hardcoding or
  # requiring an extra variable for one. A CI runner behind a proxy/VPN would need this loosened
  # (module's own documented caveat); not this reference system's posture, so left as is.
  firewall_kube_api_source = ["myipv4"]
  firewall_ssh_source      = ["myipv4"]

  # infra/scripts/burst-up.sh reads this into infra/.kube/burst-config (gitignored, the same local
  # artifact pattern infra/README.md's "Registry addressing" section already documents); false would
  # be the recommendation for an automatic CI apply, but this reference system's own `task burst:up`
  # IS the operator driven apply path, so kept true.
  create_kubeconfig = true
}
