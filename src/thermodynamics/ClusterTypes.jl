# Cluster utility accessors.

cluster_order(cluster::ThermalCluster) = cluster.order
cluster_sites(cluster::ThermalCluster) = cluster.parent_sites
cluster_bonds(cluster::ThermalCluster) = cluster.bonds

function Base.show(io::IO, c::ThermalCluster)
    print(io, "ThermalCluster(order=", c.order, ", sites=", c.parent_sites, ")")
end
