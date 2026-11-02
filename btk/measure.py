"""Module for measuring galaxy properties from images."""

from typing import Dict, List, Optional, Tuple

import galsim
import numpy as np
import sep
from astropy.table import Table
from galsim import GSObject


def _get_single_ksb_ellipticity(
    image: np.ndarray, centroid: np.ndarray, psf: GSObject, pixel_scale: float, verbose=False
) -> Tuple[float, float]:
    """Utility function to measure ellipticity using the KSB method.

    Args:
        image: Image of a single, isolated galaxy with shape (h, w).
        centroid: The centroid of the galaxy in the image, with shape (2,). Following the GalSim
            convention for offsets.
        psf: A galsim object containing the PSF of the single, isolated galaxy.
        pixel_scale: The pixel scale of the galaxy image.
        verbose: Whether to print errors if they happen when estimating ellipticity.

    Return:
        Tuple of (g1, g2) containing measured shapes.
    """
    psf_image = galsim.Image(image.shape[0], image.shape[1], scale=pixel_scale)
    gal_image = galsim.Image(image, scale=pixel_scale)
    psf_image = psf.drawImage(psf_image)
    pos = galsim.PositionD(centroid)

    res = galsim.hsm.EstimateShear(
        gal_image, psf_image, shear_est="KSB", strict=False, guess_centroid=pos
    )
    output = (res.corrected_g1, res.corrected_g2)
    if res.error_message != "":  # absorbs all (10, -10) and makes them np.nan
        output = (np.nan, np.nan)
        if verbose:
            print(
                f"Shear measurement error: '{res.error_message }'. \
                This error may happen for faint galaxies or inaccurate detections."
            )
    return output


def get_ksb_ellipticity(
    images: np.ndarray, centroids: np.ndarray, psf: GSObject, pixel_scale: float, verbose=False
) -> np.ndarray:
    """Calculate the KSB ellipticities of a batched array of isolated galaxy images.

    The galaxy images are assumed to all correspond to single band, and the input PSF is assumed
    to be the same for all images.

    If the shear measurement fails or the image is empty (no flux), then `np.nan` is returned for
    the corresponding ellipticity.

    Args:
        images: Array of batch isolated images with shape (batch_size, max_n_sources, h, w)
        centroids: An array of centers for each galaxy using the GalSim convention where the
            center of the lower-left pixel is (image.xmin, image.ymin). The shape of this array is
            (batch_size, max_n_sources, 2).
        psf: a GalSim GSObject containing the PSF common to all galaxies.
        pixel_scale: The pixel scale of the galaxy images.
        verbose: Whether an error message should be printed if the ellipticity measurement fails
            for any one of the galaxies.

    Returns:
        An array containing the measured ellipticities of shape (batch_size, max_n_sources, 2)
    """
    # psf is assumed to be the same for the entire batch and correspond to selected band.
    assert images.ndim == 4
    batch_size, max_n_sources, _, _ = images.shape
    ellipticities = np.zeros((batch_size, max_n_sources, 2))
    for ii in range(batch_size):
        for jj in range(max_n_sources):
            if np.sum(images[ii, jj]) > 0:
                ellipticities[ii, jj] = _get_single_ksb_ellipticity(
                    images[ii, jj], centroids[ii, jj], psf, pixel_scale, verbose=verbose
                )
            else:
                ellipticities[ii, jj] = (np.nan, np.nan)
    return ellipticities


def get_hsm_shapes(images: np.ndarray,centroids: np.ndarray,psf: GSObject,pixel_scale: float,
method: str = "REGAUSS",) -> Dict[str, np.ndarray]:
    """Measure galaxy shapes using GalSim's HSM code (REGAUSS by default).

    REGAUSS is the shape method the Rubin pipeline uses so I chose that one. This function returns
    shear (g1, g2) where g = (a-b) / (a+b). g1 is sideways stretch, g2 is diagonal stretch. 
    This is even for methods that measure distortion (e1, e2), so that the
    results from different methods can be compared.

    Note: If a galaxy image is empty or the measurement fails, that galaxy gets np.nan so we can skip it.

    Args:
        images: Galaxy images in a single band, shape (batch_size, max_n_sources, h, w).
        centroids: Galaxy centers (x, y), shape (batch_size, max_n_sources, 2). Uses the
            GalSim convention where the center of the lower-left pixel is (1, 1).
        psf: GalSim PSF object, assumed to be the same for all galaxies.
        pixel_scale: Pixel scale of the images in arcsec/pixel.
        method: HSM method to use: "REGAUSS", "KSB", "LINEAR", or "BJ".

    Returns:
        Dictionary with keys "g1", "g2", "resolution", "sigma". Each value is an array of
        shape (batch_size, max_n_sources).
            g1, g2: PSF-corrected shear of each galaxy.
            resolution: how resolved the galaxy is compared to the PSF (0 to 1).
            sigma: size of the galaxy in pixels.
            
    """
    
    #If someone puts a method that doesn't exist, just raise error
    if method not in ["REGAUSS", "KSB", "LINEAR", "BJ"]:
        raise ValueError(f"method must be REGAUSS, KSB, LINEAR, or BJ, not '{method}'")

    #Unpacking the dimensions of the image array
    batch_size, max_n_sources, height, width = images.shape

    # Initialize return values as arrays with size batch_size x max_n_sources with nans
    # We can't do zeros, that's a real measurement. We will fill it up later
    g1 = np.full((batch_size, max_n_sources), np.nan)
    g2 = np.full((batch_size, max_n_sources), np.nan)
    resolution = np.full((batch_size, max_n_sources), np.nan)
    sigma = np.full((batch_size, max_n_sources), np.nan)

    # HSM needs an image for the psf for later, but psf is a GalSim, so make it into one.
    psf_image = psf.drawImage(nx=width, ny=height, scale=pixel_scale)

    # For every blend
    for ii in range(batch_size):
        # For every galaxy in blend
        for jj in range(max_n_sources):
            # Skip empty slots because no galaxy in blend
            if np.sum(images[ii, jj]) <= 0:
                continue
            
            # We need an image because again, HSM accepts only images
            gal_image = galsim.Image(images[ii, jj], scale=pixel_scale)
            # Get the pixel coordinates of galaxy's center
            x, y = centroids[ii, jj]
            # Get the galaxy's shape and correct for PSF blur using aforementioned method 
            result = galsim.hsm.EstimateShear(gal_image,psf_image,shear_est=method,strict=False,
                guess_centroid=galsim.PositionD(x, y))

            # If the error message is empty, measurement succeeded. If not, then skip the galaxy
            if result.error_message != "":
                continue

            # Different methods measure different things.
            # KSB measures shear g. REGAUSS, LINEAR, and BJ measure distortion e.
            # So check which measurements we obtained
            if result.meas_type == "g":
                g1[ii, jj] = result.corrected_g1
                g2[ii, jj] = result.corrected_g2
            else:
                e1 = result.corrected_e1
                e2 = result.corrected_e2
                # abs(e) has to be less than 1, otherwise the measurement is nonsense so skip it if its true
                if e1**2 + e2**2 >= 1:
                    continue
                # Convert distortion e to shear g
                shear = galsim.Shear(e1=e1, e2=e2)
                g1[ii, jj] = shear.g1
                g2[ii, jj] = shear.g2
            # Galaxy's resolution. So 0 = low res.
            # Maybe I can forget about some galaxies based on resolution?
            resolution[ii, jj] = result.resolution_factor
            # Galaxy's size in pixels
            sigma[ii, jj] = result.moments_sigma

    return {"g1": g1, "g2": g2, "resolution": resolution, "sigma": sigma}

def get_multiband_hsm_shapes(
    images: np.ndarray,
    centroids: Optional[np.ndarray],
    psfs: Optional[list],
    pixel_scale: float,
    method: str = "REGAUSS",
) -> Dict[str, np.ndarray]:
    """Measure galaxy shapes in every band with GalSim's HSM code, using the PSF of each band.

    Observed shapes come from galsim.hsm.FindAdaptiveMom and don't need a PSF. PSF-corrected
    shapes come from galsim.hsm.EstimateShear and need the PSF of each band. If a galaxy slot is
    empty or a measurement fails, that value stays np.nan.

    For one galaxy, g1 and g2 are its ellipticity (how stretched it is and in which direction),
    written as a reduced shear. They are not the lensing shear: that only shows up when you
    average g1, g2 over many galaxies.

    The images can be anything with the right shape: isolated galaxies, deblended galaxies from
    any deblender, or the blend copied into every slot.

    Args:
        images: Galaxy images, shape (batch_size, max_n_sources, n_bands, h, w).
        centroids: Galaxy centers (x, y), shape (batch_size, max_n_sources, 2), GalSim convention
            (center of the lower-left pixel is (1, 1)). If None, HSM starts at the image center.
        psfs: List of GalSim PSF objects, one per band (like blend_batch.psf). If None, only the
            observed shapes are measured.
        pixel_scale: Pixel scale of the images in arcsec/pixel.
        method: HSM method to use: "REGAUSS", "KSB", "LINEAR", or "BJ".

    Returns:
        Dictionary where each value has shape (batch_size, max_n_sources, n_bands).
            g1_obs, g2_obs: observed ellipticity (PSF not removed, so rounder than the galaxy).
            sigma_obs: observed size in arcsec (PSF not removed).
            x_centroid, y_centroid: measured center of the galaxy in pixels (GalSim convention).
            Ixx, Iyy, Ixy: observed second moments in pixels^2 (PSF not removed).
            a, b: observed semi-major and semi-minor axes in pixels, from Ixx, Iyy, Ixy.
            g1, g2: PSF-corrected ellipticity.
            q: PSF-corrected axis ratio b/a.
            beta: PSF-corrected position angle in degrees.
            resolution: resolution factor R2 (0 = like the PSF, 1 = much bigger than the PSF).
            sigma_psf: size of the PSF in arcsec.
    """
    # If someone puts a method that doesn't exist, just raise error
    if method not in ["REGAUSS", "KSB", "LINEAR", "BJ"]:
        raise ValueError(f"method must be REGAUSS, KSB, LINEAR, or BJ, not '{method}'")

    # Unpacking the dimensions of the image array
    batch_size, max_n_sources, n_bands, height, width = images.shape

    # Start everything as nan since zero is a real measurement, we fill it up later
    keys = ["g1_obs", "g2_obs", "sigma_obs", "x_centroid", "y_centroid", "Ixx", "Iyy", "Ixy",
            "a", "b", "g1", "g2", "q", "beta", "resolution", "sigma_psf"]
    results = {}
    for key in keys:
        results[key] = np.full((batch_size, max_n_sources, n_bands), np.nan)

    # For every band
    for band in range(n_bands):
        # HSM needs an image of the PSF, so draw this band's PSF once
        psf_image = None
        if psfs is not None:
            psf_image = psfs[band].drawImage(nx=width, ny=height, scale=pixel_scale)

        # For every blend
        for ii in range(batch_size):
            # For every galaxy in blend
            for jj in range(max_n_sources):
                # Skip empty slots because no galaxy there
                if np.sum(images[ii, jj, band]) <= 0:
                    continue

                # HSM only accepts GalSim images
                gal_image = galsim.Image(images[ii, jj, band], scale=pixel_scale)

                # First guess for the galaxy's center
                guess = None
                if centroids is not None and not np.any(np.isnan(centroids[ii, jj])):
                    guess = galsim.PositionD(centroids[ii, jj, 0], centroids[ii, jj, 1])

                # Observed shape with no PSF needed
                moments = galsim.hsm.FindAdaptiveMom(gal_image, guess_centroid=guess, strict=False)
                if moments.error_message != "":
                    continue
                sigma = moments.moments_sigma
                e1 = moments.observed_shape.e1
                e2 = moments.observed_shape.e2
                # T = Ixx + Iyy
                absesq = e1**2 + e2**2
                T = 2 * sigma**2 / (1 - absesq)**0.5

                Ixx = 0.5 * T*(1+e1)
                Iyy = 0.5 * T*(1-e1)
                Ixy = 0.5 * T*e2
                
                results["Ixx"][ii, jj, band] = Ixx
                results["Iyy"][ii, jj, band] = Iyy
                results["Ixy"][ii, jj, band] = Ixy

                # Equations for a and b given Ixx, Ixy, and Iyy
                results["a"][ii, jj, band] = ((Ixx + Iyy) / 2 + ((Ixx - Iyy) ** 2 / 4 + Ixy**2) ** 0.5) ** 0.5
                results["b"][ii, jj, band] = ((Ixx + Iyy) / 2 - ((Ixx - Iyy) ** 2 / 4 + Ixy**2) ** 0.5) ** 0.5
                results["g1_obs"][ii, jj, band] = moments.observed_shape.g1
                results["g2_obs"][ii, jj, band] = moments.observed_shape.g2
                results["sigma_obs"][ii, jj, band] = moments.moments_sigma * pixel_scale
                # Where HSM found the center of the light, in pixels (GalSim convention)
                results["x_centroid"][ii, jj, band] = moments.moments_centroid.x
                results["y_centroid"][ii, jj, band] = moments.moments_centroid.y

                # Without a PSF we can't correct the shape, so stop here
                if psf_image is None:
                    continue

                # PSF-corrected shape
                result = galsim.hsm.EstimateShear(
                    gal_image, psf_image, shear_est=method, strict=False, guess_centroid=guess
                )
                if result.error_message != "":
                    continue

                # KSB measures shear g. REGAUSS, LINEAR, and BJ measure distortion e.
                if result.meas_type == "g":
                    comp1, comp2 = result.corrected_g1, result.corrected_g2
                else:
                    comp1, comp2 = result.corrected_e1, result.corrected_e2

                # e or g has to be less than 1, otherwise the measurement doesn't make sense
                if comp1**2 + comp2**2 >= 1:
                    continue

                if result.meas_type == "g":
                    shear = galsim.Shear(g1=comp1, g2=comp2)
                else:
                    shear = galsim.Shear(e1=comp1, e2=comp2)

                results["g1"][ii, jj, band] = shear.g1
                results["g2"][ii, jj, band] = shear.g2
                results["q"][ii, jj, band] = shear.q
                results["beta"][ii, jj, band] = shear.beta.deg
                results["resolution"][ii, jj, band] = result.resolution_factor
                results["sigma_psf"][ii, jj, band] = result.psf_sigma * pixel_scale
    return results

def get_blendedness(iso_image: np.ndarray) -> np.ndarray:
    """Calculate blendedness given isolated images of each galaxy in a blend.

    Args:
        iso_image: Array of shape = (..., N, H, W) corresponding to images of isolated
            galaxies you are calculating blendedness for.

    Returns:
        Array of size (..., N) corresponding to blendedness values for each individual galaxy.
    """
    assert iso_image.ndim >= 3
    num = np.sum(iso_image * iso_image, axis=(-1, -2))
    blend = np.sum(iso_image, axis=-3)[..., None, :, :]
    denom = np.sum(blend * iso_image, axis=(-1, -2))
    return 1 - np.divide(num, denom, out=np.ones_like(num), where=(num != 0))


def get_snr(iso_image: np.ndarray, sky_level: float) -> np.ndarray:
    """Calculate SNR of a set of isolated galaxies with same sky level.

    Args:
        iso_image: Array of shape = (..., H, W) corresponding to image of the isolated
            galaxy you are calculating SNR for.
        sky_level: Background level of all images. Images are assume to be
            background-substracted.

    Returns:
        Array of size (...) corresponding to SNR values for each individual galaxy.
    """
    images = iso_image + sky_level
    return np.sqrt(np.sum(iso_image * iso_image / images, axis=(-1, -2)))


def _get_single_aperture_flux(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, radius: float, sky_level: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Utility function to measure flux using fixed circular aperture with sep.

    Args:
        image (np.array): Single iamge to measure flux on, with shape (H, W).
        x (np.array): x coordinates of the center of the aperture (in pixels).
        y (np.array): y coordinates of the center of the aperture (in pixels).
        sky_level (float): Background level of all images.
            Images are assume to be background substracted.
        radius (float): Radius of the aperture in pixels.

    Returns:
        Tuple of flux and fluxerr.
    """
    assert image.ndim == 2
    assert x.ndim == 1 and y.ndim == 1
    flux, fluxerr, _ = sep.sum_circle(image, x, y, radius, var=sky_level)
    return flux, fluxerr


def get_aperture_fluxes(
    images: np.ndarray, xs: np.ndarray, ys: np.ndarray, radius: float, sky_level: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Utility function to measure flux using fixed circular aperture with sep.

    Args:
        images (np.array): Images to measure flux on, with shape (B, H, W).
        xs (np.array): x coordinates of the center of the aperture (in pixels).
        ys (np.array): y coordinates of the center of the aperture (in pixels).
        sky_level (float): Background level of all images.
            Images are assume to be background substracted.
        radius (float): Radius of the aperture in pixels.

    Returns:
        fluxes (np.array): Array of shape (B, N) corresponding to the measured aperture fluxes
            in each given position for each of the B batches.
        fluxerr (np.array): Array of same shape with corresponding flux errors.
    """
    assert images.ndim == 3
    assert xs.ndim == 2 and ys.ndim == 2
    batch_size, max_n_sources = xs.shape
    fluxes = np.zeros((batch_size, max_n_sources))
    fluxerrs = np.zeros((batch_size, max_n_sources))
    for ii in range(batch_size):
        n_sources = np.sum((xs[ii] > 0) & (ys[ii] > 0)).astype(int)
        flux, err = _get_single_aperture_flux(images[ii], xs[ii], ys[ii], radius, sky_level)
        fluxes[ii, :n_sources] = flux[:n_sources]
        fluxerrs[ii, :n_sources] = err[:n_sources]
    return fluxes, fluxerrs


def get_residual_images(iso_images: np.ndarray, blend_images: np.ndarray) -> np.ndarray:
    """Calculate residual images given isolated images of each galaxy in a blend.

    Args:
        iso_images: Array of shape = (B, N, H, W) corresponding to images of the isolated
            galaxies you are calculating residual images for.
        blend_images: Array of shape = (B, H, W) where B is the batch size. Contains noise.
    """
    except_one_images = np.sum(iso_images, axis=1)[:, None] - iso_images
    return blend_images[:, None] - except_one_images
