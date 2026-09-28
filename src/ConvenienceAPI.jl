# Cross-layer convenience constructors used by the high-level forward API.

"""
    MagneticNeutronProbe(material, cluster; site=1, prefactor=1)

Construct a magnetic-neutron probe from the magnetic form factor and g tensor
stored for one material site. This is a convenience constructor for the
existing low-level `MagneticNeutronProbe(; form_factor, g_tensor, prefactor)`
API; it does not change the scattering normalization convention.
"""
function Observables.MagneticNeutronProbe(
    material::Models.Material,
    cluster::Models.Supercell;
    site::Integer=1,
    prefactor::Real=1.0,
)
    1 <= site <= Models.nsites(cluster) || throw(BoundsError(cluster.sites, site))
    g_tensor = Models.site_g_tensor(material, cluster, site)
    isnothing(g_tensor) && throw(ArgumentError("site $site has no g tensor in Material.basis_properties"))
    magnetic_form_factor = Models.site_form_factor(material, cluster, site)
    return Observables.MagneticNeutronProbe(
        form_factor=Q -> Models.form_factor(magnetic_form_factor, Q),
        g_tensor=g_tensor,
        prefactor=prefactor,
    )
end
