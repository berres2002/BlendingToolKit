"""Contains the Deblender classes and its subclasses."""

import inspect
from abc import ABC, abstractmethod
from itertools import repeat
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import sep
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.table import Table
from numpy.linalg import LinAlgError
from skimage.feature import peak_local_max
from surveycodex.utilities import mean_sky_level

from btk.blend_batch import (
    BlendBatch,
    DeblendBatch,
    DeblendExample,
    MultiResolutionBlendBatch,
)
from btk.draw_blends import DrawBlendsGenerator
from btk.multiprocess import multiprocess


class Deblender(ABC):
    """Abstract base class containing the deblender class for BTK.

    Each new deblender class should be a subclass of Deblender.
    """

    def __init__(self, max_n_sources: int) -> None:
        """Initialize the Deblender class.

        Args:
            max_n_sources: Maximum number of sources returned by the deblender.
        """
        self.max_n_sources = max_n_sources

    def __call__(self, blend_batch: BlendBatch, njobs: int = 1, **kwargs) -> DeblendBatch:
        """Calls the (user-implemented) `deblend` method along with validation of the input.

        Args:
            ii: The index of the example in the batch.
            blend_batch: Instance of `BlendBatch` class.
            njobs: Number of processes to use.
            kwargs: Additional arguments to pass to deblender call.

        Returns:
            Instance of `DeblendedExample` class.
        """
        if not isinstance(blend_batch, BlendBatch):
            raise TypeError(
                f"Got type '{type(blend_batch)}', but expected an object of a BlendBatch class."
            )
        return self.batch_call(blend_batch, njobs, **kwargs)

    @abstractmethod
    def deblend(self, ii: int, blend_batch: BlendBatch) -> DeblendExample:
        """Runs the deblender on the ii-th example of a given batch.

        This method should be overwritten by the user if a new deblender is implemented.

        Args:
            ii: The index of the example in the batch.
            blend_batch: Instance of `BlendBatch` class

        Returns:
            Instance of `DeblendedExample` class
        """

    def batch_call(self, blend_batch: BlendBatch, njobs: int = 1, **kwargs) -> DeblendBatch:
        """Implements the call of the deblender on the entire batch.

        Overwrite this function if you perform measurments on the batch.
        The default fucntionality is to use multiprocessing to speed up
        the iteration over all examples in the batch.

        Args:
            blend_batch: Instance of `BlendBatch` class
            njobs: Number of jobs to paralelize across
            kwargs: Additional keyword arguments to pass to each deblend call.

        Returns:
            Instance of `DeblendedBatch` class
        """
        args_iter = ((ii, blend_batch) for ii in range(blend_batch.batch_size))
        kwargs_iter = repeat(kwargs)
        output = multiprocess(self.deblend, args_iter, kwargs_iter, njobs=njobs)
        catalog_list = [db_example.catalog for db_example in output]
        segmentation, deblended, extra_data = None, None, None
        n_bands = None
        if output[0].segmentation is not None:
            segmentation = np.array([db_example.segmentation for db_example in output])
        if output[0].deblended_images is not None:
            deblended = np.array([db_example.deblended_images for db_example in output])
            _, _, n_bands, _, _ = deblended.shape
        if output[0].extra_data is not None:
            extra_data = [db_example.extra_data for db_example in output]
        return DeblendBatch(
            blend_batch.batch_size,
            self.max_n_sources,
            catalog_list,
            n_bands,
            blend_batch.image_size,
            segmentation,
            deblended,
            extra_data,
        )

    @classmethod
    def __repr__(cls):
        """Returns the name of the class for bookkeeping."""
        return cls.__name__


class MultiResolutionDeblender(ABC):
    """Abstract base class for deblenders using multiresolution images."""

    def __init__(self, max_n_sources: int, survey_names: Union[list, tuple]) -> None:
        """Initialize the multiresolution deblender."""
        assert isinstance(survey_names, (list, tuple))
        assert len(survey_names) > 1, "At least two surveys must be used."

        self.max_n_sources = max_n_sources
        self.survey_names = survey_names

    def __call__(
        self, ii: int, mr_batch: MultiResolutionBlendBatch, njobs: int = 1
    ) -> DeblendBatch:
        """Calls the (user-implemented) deblend method along with validation of the input.

        Args:
            ii: The index of the example in the batch.
            mr_batch: Instance of `MultiResolutionBlendBatch` class
            njobs: Number of processes to use.

        Returns:
            Instance of `DeblendedExample` class
        """
        if not isinstance(mr_batch, MultiResolutionBlendBatch):
            raise TypeError(
                f"Got type'{type(mr_batch)}', but expected a MultiResolutionBlendBatch object."
            )
        self.batch_call(mr_batch, njobs)

    @abstractmethod
    def deblend(self, ii: int, mr_batch: MultiResolutionBlendBatch) -> DeblendExample:
        """Runs the MR deblender on the ii-th example of a given batch.

        This method should be overwritten by the user if a new deblender is implemented.

        Args:
            ii: The index of the example in the batch.
            mr_batch: Instance of `MultiResolutionBlendBatch` class

        Returns:
            Instance of `DeblendedExample` class
        """

    def batch_call(self, mr_batch: MultiResolutionBlendBatch, njobs: int = 1) -> DeblendBatch:
        """Implements the call of the deblender on the entire batch.

        Overwrite this function if you perform measurments on a batch.
        The default fucntionality is to use multiprocessing to speed up
        the iteration over all examples in the batch.

        Args:
            mr_batch: Instance of `MultiResolutionBlendBatch` class
            njobs: Number of njobs to paralelize across

        Returns:
            Instance of `DeblendedBatch` class
        """
        raise NotImplementedError("No multi-resolution deblender has been implemented yet.")

    @classmethod
    def __repr__(cls):
        """Returns the name of the class for bookkeeping."""
        return cls.__name__


class PeakLocalMax(Deblender):
    """This class detects centroids with `skimage.feature.peak_local_max`.

    The function performs detection and deblending of the sources based on the provided
    band index. If use_mean feature is used, then the Deblender will use
    the average of all the bands.
    """

    def __init__(
        self,
        max_n_sources: int,
        sky_level: float,
        threshold_scale: int = 5,
        min_distance: int = 2,
        use_mean: bool = False,
        use_band: Optional[int] = None,
    ) -> None:
        """Initializes Deblender class. Exactly one of 'use_mean' or 'use_band' must be specified.

        Args:
            max_n_sources: See parent class.
            sky_level: Background intensity in images to be detected (assumed constant).
            threshold_scale: Minimum number of sigmas above noise level for detections.
            min_distance: Minimum distance in pixels between two peaks.
            use_mean: Flag to use the band average for deblending.
            use_band: Integer index of the band to use for deblending
        """
        super().__init__(max_n_sources)
        self.min_distance = min_distance
        self.threshold_scale = threshold_scale
        self.sky_level = sky_level

        if use_band is None and not use_mean:
            raise ValueError("Either set 'use_mean=True' OR indicate a 'use_band' index")
        if use_band is not None and use_mean:
            raise ValueError("Only one of the parameters 'use_band' and 'use_mean' has to be set")
        self.use_mean = use_mean
        self.use_band = use_band

    def deblend(self, ii: int, blend_batch: BlendBatch) -> DeblendExample:
        """Performs deblending on the ii-th example from the batch."""
        blend_image = blend_batch.blend_images[ii]
        image = np.mean(blend_image, axis=0) if self.use_mean else blend_image[self.use_band]

        # compute threshold value
        threshold = self.threshold_scale * np.sqrt(self.sky_level)

        # calculate coordinates
        coordinates = peak_local_max(
            image,
            min_distance=self.min_distance,
            threshold_abs=threshold,
            num_peaks=self.max_n_sources,
        )
        x, y = coordinates[:, 1], coordinates[:, 0]

        # convert coordinates to ra, dec
        wcs = blend_batch.wcs
        ra, dec = wcs.pixel_to_world_values(x, y)
        ra *= 3600
        dec *= 3600

        # wrap in catalog
        catalog = Table()
        catalog["ra"], catalog["dec"] = ra, dec
        catalog["x_peak"], catalog["y_peak"] = x, y

        if len(catalog) > self.max_n_sources:
            raise ValueError(
                "`PeakLocalMax` detected more sources than `max_n_sources`. Consider increasing"
                "`threshold_scale` or `max_n_sources`."
            )

        return DeblendExample(self.max_n_sources, catalog)


class SepSingleBand(Deblender):
    """Return detection, segmentation and deblending information running SEP on a single band.

    The function performs detection and deblending of the sources based on the provided
    band index. If `use_mean` feature is used, then we use the average of all the bands.

    For more details on SEP (Source-Extractor Python), see:
    https://sep.readthedocs.io/en/v1.1.x/index.html#
    """

    def __init__(
        self,
        max_n_sources: int,
        thresh: float = 1.5,
        min_area: int = 5,
        use_mean: bool = False,
        use_band: Optional[int] = None,
    ) -> None:
        """Initializes Deblender class. Exactly one of 'use_mean' or 'use_band' must be specified.

        Args:
            max_n_sources: See parent class.
            thresh: Threshold pixel value for detection use in `sep.extract`. This is
                interpreted as a relative threshold: the absolute threshold at pixel (j, i)
                will be `thresh * err[j, i]` where `err` is set to the global rms of
                the background measured by SEP.
            min_area: Minimum number of pixels required for an object. Default is 5.
            use_mean: Flag to use the band average for deblending.
            use_band: Integer index of the band to use for deblending.
        """
        super().__init__(max_n_sources)
        if use_band is None and not use_mean:
            raise ValueError("Either set 'use_mean=True' OR indicate a 'use_band' index")
        if use_band is not None and use_mean:
            raise ValueError("Only one of the parameters 'use_band' and 'use_mean' has to be set")
        self.use_mean = use_mean
        self.use_band = use_band
        self.thresh = thresh
        self.min_area = min_area

    def deblend(self, ii: int, blend_batch: BlendBatch) -> DeblendExample:
        """Performs deblending on the i-th example from the batch."""
        # get a 1-channel input for sep
        blend_image = blend_batch.blend_images[ii]
        image = np.mean(blend_image, axis=0) if self.use_mean else blend_image[self.use_band]

        # run source extractor
        bkg = sep.Background(image)
        catalog, segmentation = sep.extract(
            image,
            self.thresh,
            err=bkg.globalrms,
            segmentation_map=True,
            minarea=self.min_area,
        )

        if len(catalog) > self.max_n_sources:
            raise ValueError(
                "SEP predicted more sources than `max_n_sources`. Consider increasing `thresh`"
                " or `max_n_sources`."
            )

        segmentation_exp = np.zeros((self.max_n_sources, *image.shape), dtype=bool)
        deblended_images = np.zeros((self.max_n_sources, *image.shape), dtype=image.dtype)
        n_objects = len(catalog)
        for jj in range(n_objects):
            seg_i = segmentation == jj + 1
            segmentation_exp[jj] = seg_i
            deblended_images[jj] = image * seg_i.astype(image.dtype)

        # convert to ra, dec
        wcs = blend_batch.wcs
        ra, dec = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
        ra *= 3600
        dec *= 3600

        # wrap results in astropy table
        cat = Table()
        cat["ra"], cat["dec"] = ra, dec
        cat["x_peak"], cat["y_peak"] = catalog["x"], catalog["y"]

        return DeblendExample(
            self.max_n_sources,
            cat,
            1,  # single band is returned
            blend_batch.image_size,
            segmentation_exp,
            deblended_images[:, None].clip(0),  # add a channel dimension
        )


class SepMultiBand(Deblender):
    """This class returns centers detected with SEP by combining predictions in different bands.

    For each band in the input image we run `sep` for detection and append new detections
    to a running list of detected coordinates. In order to avoid repeating detections,
    we run a KD-Tree algorithm to calculate the angular distance between each new
    coordinate and its closest neighbour. Then we discard those new coordinates that
    were closer than `matching_threshold` to any one of already detected coordinates.
    """

    def __init__(self, max_n_sources: int, matching_threshold: float = 1.0, thresh: float = 1.5):
        """Initialize the SepMultiBand Deblender.

        Args:
            max_n_sources: See parent class.
            matching_threshold: Threshold value for match detections that are close (arcsecs).
            thresh: See `SepSingleBand` class.
        """
        super().__init__(max_n_sources)
        self.matching_threshold = matching_threshold
        self.thresh = thresh

    def deblend(self, ii: int, blend_batch: BlendBatch) -> DeblendExample:
        """Performs deblending on the ii-th example from the batch."""
        # run source extractor on the first band
        wcs = blend_batch.wcs
        image = blend_batch.blend_images[ii]
        bkg = sep.Background(image[0])
        catalog = sep.extract(image[0], self.thresh, err=bkg.globalrms, segmentation_map=False)
        ra_coordinates, dec_coordinates = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
        ra_coordinates *= 3600
        dec_coordinates *= 3600

        if len(catalog) > self.max_n_sources:
            raise ValueError(
                "SEP predicted more sources than `max_n_sources`. Consider increasing `thresh`"
                " or `max_n_sources`."
            )

        # iterate over remaining bands and match predictions using KdTree
        for band in range(1, image.shape[0]):
            # run source extractor
            band_image = image[band]
            bkg = sep.Background(band_image)
            catalog = sep.extract(
                band_image, self.thresh, err=bkg.globalrms, segmentation_map=False
            )

            if len(catalog) > self.max_n_sources:
                raise ValueError(
                    "SEP predicted more sources than `max_n_sources`. Consider increasing `thresh`"
                    " or `max_n_sources`."
                )

            # convert predictions to arcseconds
            ra_detections, dec_detections = wcs.pixel_to_world_values(catalog["x"], catalog["y"])
            ra_detections *= 3600
            dec_detections *= 3600

            # convert to sky coordinates
            c1 = SkyCoord(ra=ra_detections * units.arcsec, dec=dec_detections * units.arcsec)
            c2 = SkyCoord(ra=ra_coordinates * units.arcsec, dec=dec_coordinates * units.arcsec)

            # merge new detections with the running list of coordinates
            if len(c1) > 0 and len(c2) > 0:
                # runs KD-tree to get distances to the closest neighbours
                _, distance2d, _ = c1.match_to_catalog_sky(c2)
                distance2d = distance2d.arcsec

                # add new predictions, masking those that are closer than threshold
                ra_coordinates = np.concatenate(
                    [
                        ra_coordinates,
                        ra_detections[distance2d > self.matching_threshold],
                    ]
                )
                dec_coordinates = np.concatenate(
                    [
                        dec_coordinates,
                        dec_detections[distance2d > self.matching_threshold],
                    ]
                )
            else:
                ra_coordinates = np.concatenate([ra_coordinates, ra_detections])
                dec_coordinates = np.concatenate([dec_coordinates, dec_detections])

        # Wrap in the astropy table
        catalog = Table()
        catalog["ra"] = ra_coordinates
        catalog["dec"] = dec_coordinates
        return DeblendExample(self.max_n_sources, catalog)

class DeepDisc(Deblender):

    def __init__(self, max_n_sources: int, model_path: str, config_path: str, score_thresh: float = 0.3, nms_thresh: float = 0.5, allow_over_n_sources: bool = False):
        from detectron2.config import get_cfg
        import deepdisc.astrodet.astrodet as toolkit
        # from deepdisc.astrodet import detectron as detectron_addons
        from detectron2.config import LazyConfig
        import os
        super().__init__(max_n_sources)
        self.model_path = model_path
        # Add path to config file
        self.config_path = config_path

        # Add threshold for detection
        # self.topk_per_image = 3000
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.allow_over_n_sources = allow_over_n_sources
        #load cfg
        cfgfile = self.config_path # determine later
        self.cfg = LazyConfig.load(cfgfile)
        self.cfg.OUTPUT_DIR = './'
        self.cfg.train.init_checkpoint = os.path.join(self.cfg.OUTPUT_DIR, self.model_path) # <- Path to model weights, change this to trained model weights
        for box_predictor in self.cfg.model.roi_heads.box_predictors:
            # box_predictor.test_topk_per_image = 3000
            box_predictor.test_score_thresh = self.score_thresh
            box_predictor.test_nms_thresh = self.nms_thresh
        # Load predictor function
        self.predictor = toolkit.AstroPredictor(self.cfg) # Predictor class makes the predictions

    def deblend(self, ii: int, blend_batch: BlendBatch) -> DeblendExample:
        # import os

        from astropy.visualization import make_lupton_rgb
        from PIL import Image, ImageEnhance

        

        # from deepdisc.data_format.file_io import DDLoader
        # from deepdisc.data_format.annotation_functions.annotate_decam import annotate_decam

        # Setup cfg
        # cfg = get_cfg()

        # Load model
        # cfgfile = self.config_path # determine later
        # cfg = LazyConfig.load(cfgfile)
        # cfg.OUTPUT_DIR = './'
        # cfg.train.init_checkpoint = os.path.join(cfg.OUTPUT_DIR, self.model_path) # <- Path to model weights, change this to trained model weights

        #change these to play with the detection sensitivity
        # for box_predictor in self.cfg.model.roi_heads.box_predictors:
        #     # box_predictor.test_topk_per_image = 3000
        #     box_predictor.test_score_thresh = self.score_thresh
        #     box_predictor.test_nms_thresh = self.nms_thresh
        # reader_scaling = cfg.dataloader.imagereader.scaling
        def read_image_hsc(ii,
            normalize="lupton",
            stretch=0.5,
            Q=10,
            m=0,
            ceil_percentile=99.995,
            dtype=np.uint8,
            A=1e4,
            do_norm=False,
            ):
            """
            Read in a formatted HSC image

            Parameters
            ----------
            filenames: list
                The list of g,r,i band files
            normalize: str
                The key word for the normalization scheme
            stretch, Q, m: float, int, float
                Parameters for lupton normalization
            ceil_percentile:
                If do_norm is true, cuts data off at this percentile
            dtype: numpy datatype
                data type of the output array
            A: float
                scaling factor for zscoring
            do_norm: boolean
                For normalizing top fit dtype range

            Returns
            -------
            Scaled image

            """

            def norm(z, r, g):
                max_RGB = np.nanpercentile([z, r, g], ceil_percentile)
                print(max_RGB)

                max_z = np.nanpercentile([z], ceil_percentile)
                max_r = np.nanpercentile([r], ceil_percentile)
                max_g = np.nanpercentile([g], ceil_percentile)

                # z = np.clip(z,None,max_RGB)
                # r = np.clip(r,None,max_RGB)
                # g = np.clip(g,None,max_RGB)

                # avoid saturation
                r = r / max_RGB
                g = g / max_RGB
                z = z / max_RGB
                # r = r/max_r; g = g/max_g; z = z/max_z

                # Rescale to 0-255 for dtype=np.uint8
                max_dtype = np.iinfo(dtype).max
                r = r * max_dtype
                g = g * max_dtype
                z = z * max_dtype

                # 0-255 RGB image
                image[:, :, 0] = z  # R
                image[:, :, 1] = r  # G
                image[:, :, 2] = g  # B

                return image

            # Read image

            # need to figure out which indices are which bands
            g = blend_batch.blend_images[ii][0]
            r = blend_batch.blend_images[ii][1]
            z = blend_batch.blend_images[ii][3]

            # Contrast scaling / normalization
            I = (z + r + g) / 3.0

            length, width = g.shape
            image = np.empty([length, width, 3], dtype=dtype)

            # asinh(Q (I - minimum)/stretch)/Q

            # Options for contrast scaling
            if normalize.lower() == "lupton" or normalize.lower() == "luptonhc":
                z = z * np.arcsinh(stretch * Q * (I - m)) / (Q * I)
                r = r * np.arcsinh(stretch * Q * (I - m)) / (Q * I)
                g = g * np.arcsinh(stretch * Q * (I - m)) / (Q * I)

                # z = z*np.arcsinh(Q*(I - m)/stretch)/(Q)
                # r = r*np.arcsinh(Q*(I - m)/stretch)/(Q)
                # g = g*np.arcsinh(Q*(I - m)/stretch)/(Q)
                image[:, :, 0] = z  # R
                image[:, :, 1] = r  # G
                image[:, :, 2] = g  # B
                if do_norm:
                    return norm(z, r, g)
                else:
                    return image

            elif normalize.lower() == "astrolupton":
                image = make_lupton_rgb(z, r, g, minimum=m, stretch=stretch, Q=Q)
                return image

            elif normalize.lower() == "zscore":
                Imean = np.nanmean(I)
                Isigma = np.nanstd(I)

                z = A * (z - Imean - m) / Isigma
                r = A * (r - Imean - m) / Isigma
                g = A * (g - Imean - m) / Isigma

                image[:, :, 0] = z  # R
                image[:, :, 1] = r  # G
                image[:, :, 2] = g  # B
                if do_norm:
                    return norm(z, r, g)
                else:
                    return image

            elif normalize.lower() == "zscore_orig":
                zsigma = np.nanstd(z)
                rsigma = np.nanstd(r)
                gsigma = np.nanstd(g)

                z = A * (z - np.nanmean(z) - m) / zsigma
                r = A * (r - np.nanmean(r) - m) / rsigma
                g = A * (g - np.nanmean(g) - m) / gsigma

                image[:, :, 0] = z  # R
                image[:, :, 1] = r  # G
                image[:, :, 2] = g  # B

                return image

            elif normalize.lower() == "sinh":
                z = np.sinh((z - m))
                r = np.sinh((r - m))
                g = np.sinh((g - m))

            # sqrt(Q (I - minimum)/stretch)/Q
            elif normalize.lower() == "sqrt":
                z = z * np.sqrt((I - m) * Q / stretch) / I / stretch
                r = r * np.sqrt((I - m) * Q / stretch) / I / stretch
                g = g * np.sqrt((I - m) * Q / stretch) / I / stretch
                image[:, :, 0] = z  # R
                image[:, :, 1] = r  # G
                image[:, :, 2] = g  # B
                if do_norm:
                    return norm(z, r, g)
                else:
                    return image

            elif normalize.lower() == "sqrt-old":
                z = np.sqrt(z)
                r = np.sqrt(r)
                g = np.sqrt(g)
                image[:, :, 0] = z  # R
                image[:, :, 1] = r  # G
                image[:, :, 2] = g  # B
                if do_norm:
                    return norm(z, r, g)
                else:
                    return image

            elif normalize.lower() == "linear":
                z = A * (z - m)
                r = A * (r - m)
                g = A * (g - m)
                # z = (z - m)
                # r = (r - m)
                # g = (g - m)

                image[:, :, 0] = z  # R
                image[:, :, 1] = r  # G
                image[:, :, 2] = g  # B
                return image

            elif normalize.lower() == "normlinear":
                # image = np.empty([length, width, 3], dtype=dtype)

                z = A * (z - m)
                r = A * (r - m)
                g = A * (g - m)
                # z = (z - m)
                # r = (r - m)
                # g = (g - m)

                # image[:,:,0] = z # R
                # image[:,:,1] = r # G
                # image[:,:,2] = g # B
                # return image

            elif normalize.lower() == "astroluptonhc":
                image = make_lupton_rgb(z, r, g, minimum=m, stretch=stretch, Q=Q)
                factor = 2  # gives original image
                cenhancer = ImageEnhance.Contrast(Image.fromarray(image))
                im_output = cenhancer.enhance(factor)
                benhancer = ImageEnhance.Brightness(im_output)
                image = benhancer.enhance(factor)
                image = np.asarray(image)
                return image

            else:
                print("Normalize keyword not recognized.")




        # Load image
        # img =  read_image_hsc(ii)
        def reader_find():
            try:
                reader = self.cfg.dataloader.imagereader
            except:
                from deepdisc.data_format.image_readers import RomanImageReader
                reader = RomanImageReader()
            return reader
        def reader_save(bb):
            reader = reader_find()
            img = reader(bb)
            return img
        img = reader_save(blend_batch.blend_images[ii])
        # Make inference

        output = self.predictor(img)

        # Get centers of Bounding Boxes 

        # Create dict for extra data, has to be None if not populated
        extra_dict = {}

        centers = output['instances'].pred_boxes.get_centers().cpu().numpy()

        wcs = blend_batch.wcs

        ra_coordinates, dec_coordinates = wcs.pixel_to_world_values(centers[:,0], centers[:,1])
        ra_coordinates *= 3600
        dec_coordinates *= 3600

        catalog = Table()
        catalog["ra"] = ra_coordinates
        catalog["dec"] = dec_coordinates
        catalog['x_peak'] = centers[:,0]
        catalog['y_peak'] = centers[:,1]
        # Place in DeblendExample object

        # catalog = 'HSC' # change this later

        # add segmentation and deblended images
        segmentation = output["instances"].pred_masks.cpu().numpy()
        
        # print(segmentation.shape,self.max_n_sources)
        if segmentation.shape[0] > self.max_n_sources and self.allow_over_n_sources == False:
            raise ValueError(
                "DeepDISC predicted more sources than `max_n_sources`. Consider increasing `score_thresh`"
                " or decreasing `max_n_sources`."
                f"Detections {segmentation.shape[0]} > {self.max_n_sources}"
            )
        elif segmentation.shape[0] > self.max_n_sources and self.allow_over_n_sources == True:
            print("DeepDISC predicted more sources than `max_n_sources`. Allowing the output of more sources "
            "than `max_n_sources` as"
            " `allow_over_n_sources` was set to `True`.")
            # self.max_n_sources = segmentation.shape[0]
            segs = np.zeros((self.max_n_sources, img.shape[0], img.shape[1]),dtype=segmentation.dtype) #create empty array
            segs[:self.max_n_sources] = segmentation[:self.max_n_sources] # fill segs array UP to max_n_sources
            extra_segs = segmentation[(self.max_n_sources-segmentation.shape[0]):] # get the rest of the segmentation maps
            extra_dict = {'segs': extra_segs} # add to extra_dict

            deblended_images = np.zeros((segmentation.shape[0], img.shape[2],img.shape[0],img.shape[1]))
            rimg = np.transpose(img,(2,0,1))
            for i in range(segmentation.shape[0]):
                deblended_images[i] = rimg * segmentation[i].astype(img.dtype) # cuts out source from image using segmentation mask
            deblended_ims = deblended_images[:self.max_n_sources] # only keep the first max_n_sources
            extra_deblend = deblended_images[(self.max_n_sources-segmentation.shape[0]):] # get the rest of the deblended images
            extra_dict['deblended_images'] = extra_deblend # add to extra_dict
        else:
            segs = np.zeros((self.max_n_sources, img.shape[0], img.shape[1]),dtype=segmentation.dtype)
            segs[:segmentation.shape[0]] = segmentation
            deblended_images = np.zeros((self.max_n_sources, img.shape[2],img.shape[0],img.shape[1]))
            rimg = np.transpose(img,(2,0,1))
            for i in range(segmentation.shape[0]):
                deblended_images[i] = rimg * segmentation[i].astype(img.dtype) # cuts out source from image using segmentation mask
            deblended_ims = deblended_images
        # OR outputs.sem_seg, outputs.panoptic_seg
        return DeblendExample(self.max_n_sources,
                    catalog,
                    img.shape[2],
                    blend_batch.image_size,
                    segs,
                    deblended_ims,
                    extra_dict)
    
class Scarlet(Deblender):
    """Implementation of the scarlet deblender."""

    def __init__(
        self,
        max_n_sources: int,
        thresh: float = 1.0,
        e_rel: float = 1e-5,
        max_iter: int = 200,
        max_components: int = 2,
        min_snr: float = 50,
    ):
        """Initialize the Scarlet deblender class.

        This class uses the scarlet deblender to deblend the images. We follow the basic
        implementation that is layed out in the Scarlet documentation:
        https://pmelchior.github.io/scarlet/0-quickstart.html. For more details on each
        of the argument also see the Scarlet API at:
        https://pmelchior.github.io/scarlet/api/scarlet.initialization.html.

        Note that as of commit 45187fd, Scarlet raises a `LinAlg` error if two sources are on
        the same pixel, which is allowed by the majority of currently implemented sampling
        functions in BTK. To get around this, our Deblender implementation automatically
        catches this exception and outputs an array of zeroes for the deblended images of the
        particular blend that caused this exception. See this issue for details:
        https://github.com/pmelchior/scarlet/issues/282#issuecomment-2074886534

        Args:
            max_n_sources: See parent class.
            thresh: Multiple of the backround RMS used as a flux cutoff for morphology
                initialization for `scarlet.source.ExtendedSource` class. (Default: 1.0)
            e_rel: Relative error for convergence of the loss function
                See `scarlet.blend.Blend.fit` method for details. (Default: 1e-5)
            max_iter: Maximum number of iterations for the optimization (Default: 200)
            min_snr: Mininmum SNR per component to accept the source. (Default: 50)
            max_components: Maximum number of components in a source. (Default: 2)
        """
        super().__init__(max_n_sources)
        self.thres = thresh
        self.e_rel = e_rel
        self.max_iter = max_iter
        self.max_components = max_components
        self.min_snr = min_snr

    def deblend(
        self, ii: int, blend_batch: BlendBatch, reference_catalogs: Table = None
    ) -> DeblendExample:
        """Performs deblending on the ii-th example from the batch.

        Args:
            ii: The index of the example in the batch.
            blend_batch: Instance of `BlendBatch` class.
            reference_catalogs: Reference catalog to use for deblending. If None, the
                truth catalog is used.

        Returns:
            Instance of `DeblendedExample` class.
        """
        import scarlet  # pylint: disable=import-outside-toplevel

        # if no reference catalog is provided, truth catalog is used
        if reference_catalogs is None:
            catalog = blend_batch.catalog_list[ii]
        else:
            catalog = reference_catalogs[ii]
        assert "ra" in catalog.colnames and "dec" in catalog.colnames

        image = blend_batch.blend_images[ii]
        n_bands = image.shape[0]

        psf = blend_batch.get_numpy_psf()
        wcs = blend_batch.wcs
        survey = blend_batch.survey
        bands = survey.available_filters
        img_size = blend_batch.image_size

        # if catalog is empty return no images.
        if len(catalog) == 0:
            t = Table()
            t["ra"] = []
            t["dec"] = []
            deblended_images = np.zeros((self.max_n_sources, n_bands, img_size, img_size))
            return DeblendExample(self.max_n_sources, t, n_bands, img_size, None, deblended_images)

        # get background
        filters = [survey.get_filter(band) for band in bands]
        bkg = np.array([mean_sky_level(survey, f).to_value("electron") for f in filters])

        # initialize scarlet
        model_psf = scarlet.GaussianPSF(sigma=(0.6,) * n_bands)
        model_frame = scarlet.Frame(image.shape, psf=model_psf, channels=bands, wcs=wcs)
        scarlet_psf = scarlet.ImagePSF(psf)
        weights = np.ones(image.shape) / bkg.reshape((-1, 1, 1))
        obs = scarlet.Observation(image, psf=scarlet_psf, weights=weights, channels=bands, wcs=wcs)
        observations = obs.match(model_frame)

        ra_dec = np.array([catalog["ra"] / 3600, catalog["dec"] / 3600]).T

        try:
            sources, _ = scarlet.initialization.init_all_sources(
                model_frame,
                ra_dec,
                observations,
                max_components=self.max_components,
                min_snr=self.min_snr,
                thresh=self.thres,
                fallback=True,
                silent=True,
                set_spectra=True,
            )

            blend = scarlet.Blend(sources, observations)
            blend.fit(self.max_iter, e_rel=self.e_rel)
            individual_sources, selected_peaks = [], []
            for component in sources:
                y, x = component.center
                selected_peaks.append([x, y])
                model = component.get_model(frame=model_frame)
                model_ = observations.render(model)
                individual_sources.append(model_)
            selected_peaks = np.array(selected_peaks)
            deblended_images = np.zeros((self.max_n_sources, n_bands, img_size, img_size))
            deblended_images[: len(individual_sources)] = individual_sources

            assert len(selected_peaks) == len(catalog)

            return DeblendExample(
                self.max_n_sources,
                catalog,
                n_bands,
                blend_batch.image_size,
                None,
                deblended_images.clip(0),  # rarely scarlet gives very small neg. values
                extra_data={"scarlet_sources": sources},
            )

        except LinAlgError:
            t = Table()
            t["ra"] = []
            t["dec"] = []
            deblended_images = np.zeros((self.max_n_sources, n_bands, img_size, img_size))
            return DeblendExample(self.max_n_sources, t, n_bands, img_size, None, deblended_images)


class DeblendGenerator:
    """Run one or more deblenders on the batches from the given draw_blend_generator."""

    def __init__(
        self,
        deblenders: Union[List[Deblender], Deblender],
        draw_blend_generator: DrawBlendsGenerator,
        njobs: int = 1,
        verbose: bool = False,
    ):
        """Initialize deblender generator.

        Args:
            deblenders: Deblender or a list of Deblender that will be used on the
                            outputs of the draw_blend_generator.
            draw_blend_generator: Instance of subclasses of `DrawBlendsGenerator`.
            njobs: The number of parallel processes to run [Default: 1].
            verbose: Whether to print information about deblending.
        """
        self.deblenders = self._validate_deblenders(deblenders)
        self.deblender_names = self._get_unique_deblender_names()
        self.draw_blend_generator = draw_blend_generator
        self.njobs = njobs

        self.batch_size = self.draw_blend_generator.batch_size
        self.verbose = verbose

    def __iter__(self):
        """Return iterator which is the object itself."""
        return self

    def _validate_deblenders(self, deblenders) -> List[Deblender]:
        """Ensure all measure functions are subclasses of `Measure` and correctly instantiated."""
        if not isinstance(deblenders, list):
            deblenders = [deblenders]

        for deblender in deblenders:
            if inspect.isclass(deblender):
                is_deblender_cls = issubclass(deblender, Deblender)
                is_mr_deblender_cls = issubclass(deblender, MultiResolutionDeblender)
                if not (is_deblender_cls or is_mr_deblender_cls):
                    raise TypeError(
                        f"'{deblender.__name__}' must subclass from Deblender or"
                        "MultiResolutionDeblender."
                    )
                raise TypeError(
                    f"'{deblender.__name__}' must be instantiated. Use '{deblender.__name__}()'"
                )

            is_deblender = isinstance(deblender, Deblender)
            is_mr_deblender = isinstance(deblender, MultiResolutionDeblender)
            if not is_deblender and not is_mr_deblender:
                raise TypeError(
                    f"Got type'{type(deblender)}', but expected an object of a Deblender or"
                    "MultiResolutionDeblender class."
                )
        return deblenders

    def _get_unique_deblender_names(self) -> List[str]:
        """Get list of unique indexed names of each deblender passed in (as necessary)."""
        deblender_names = [str(deblender) for deblender in self.deblenders]
        names_counts = {name: 0 for name in set(deblender_names) if deblender_names.count(name) > 1}
        for ii, name in enumerate(deblender_names):
            if name in names_counts:
                deblender_names[ii] += f"_{names_counts[name]}"
                names_counts[name] += 1
        return deblender_names

    def __next__(self) -> Tuple[BlendBatch, Dict[str, DeblendBatch]]:
        """Return deblending results on a single batch from the draw_blend_generator.

        Returns:
            blend_batch: draw_blend_generator output from its `__next__` method.
            deblended_output (dict): Dictionary with keys being the name of each
                measure function passed in, and each value its corresponding `MeasuredBatch`.
        """
        blend_batch = next(self.draw_blend_generator)
        deblended_output = {
            name: deblender.batch_call(blend_batch, njobs=self.njobs)
            for name, deblender in zip(self.deblender_names, self.deblenders)
        }
        return blend_batch, deblended_output


available_deblenders = {
    "PeakLocalMax": PeakLocalMax,
    "SepSingleBand": SepSingleBand,
    "SepMultiBand": SepMultiBand,
    "Scarlet": Scarlet,
}
