from pydantic import BaseModel


class EquilibriumData(BaseModel):
    """
    MHD equilibrium reconstruction data for a plasma shot.

    time_s: time points of the equilibrium reconstruction
    psi_norm: normalised poloidal flux grid (0=axis, 1=LCFS)
    r_m / z_m: R-Z grid for 2-D flux surface maps (optional)
    q_profile: safety factor profile on psi_norm grid
    """

    shot_id: str
    reconstruction_code: str | None = None  # e.g. "EFIT", "VMEC"
    time_s: list[float]
    psi_norm: list[float]
    r_m: list[float] | None = None
    z_m: list[float] | None = None
    q_profile: list[float] | None = None
    ip_MA: float | None = None
    beta_total: float | None = None
    beta_poloidal: float | None = None
    li: float | None = None
    r_major_m: float | None = None
    a_minor_m: float | None = None
    kappa: float | None = None
    delta: float | None = None
    metadata: dict = {}
