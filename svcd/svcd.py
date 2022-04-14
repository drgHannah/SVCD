"""SVCD Algorithm.

This module inlcudes an unofficial python implementation 
of the segmentation algorithm proposed by Nieuwenhuis and Cremers in 2013:
"Spatially Varying Color Distributions for Interactive Multilabel Segmentation"

Authors:
    Markus Plack <markus.plack@gmail.com>
    Hannah Dröge <drg.hannah@gmail.com>
"""
import logging
from typing import Optional

import numpy as np
from PIL import Image
from scipy import sparse
try:
    tqdm = None
    from tqdm import tqdm
except ImportError:
    pass

class SVCDutils():
    """Helper Functions for Segmentation

    Provides methods for both loading an image and 
    its scribbles in the correct format, and saving 
    the segmentation with the selcted color space.
    """

    @staticmethod
    def load_image(fpath):
        """Load the image as numpy array.
    
        Args:
            fpath: path of an RGB image

        Returns:
            A numpy array of the loaded image.
        """
        image = Image.open(fpath)
        array = np.asarray(image) / 255
        array = np.transpose(array, (2, 0, 1))
        return array

    @staticmethod
    def load_segmentation_from_image(fpath, max_classes=16):
        """Load an image as a segmentation/scribble

        Each unique color is treated as a class.
        Black values as well as pixels with alpha<1 are assumed to be unkown.

        Args:
            fpath: path of an RGB image
            max_classes: the maximum number of classes

        Returns:
            A segmentation of shape [class, height, width] and a list of colors.

        Raises:
            IOError: An error occurred reading the classes form the scribbles.
        """
        # Load the image
        image = Image.open(fpath)
        segmentation_image = np.asarray(image) / 255
        segmentation_image = np.transpose(segmentation_image, (2, 0, 1))
        # Remove alpa channel (if exists)
        if segmentation_image.shape[0] == 4:
            alpha = segmentation_image[3]
            segmentation_image = segmentation_image[:3]
            segmentation_image[:, alpha < 1] = 0
        # Find unique colors and create masks
        colors = []
        masks = []
        nonzero = np.nonzero(segmentation_image)
        while len(nonzero[0]) > 0:
            if len(colors) >= max_classes:
                logging.getLogger(__name__).warning(
                    f"Image '{fpath}' contains too many classes. "
                    f"Limiting classes to {max_classes}. "
                    f"{len(nonzero[0])} pixels left as background."
                )
            color = segmentation_image[:, nonzero[1][0], nonzero[2][0]]
            colors.append(color.copy())
            mask = np.all(color.reshape((3,1,1)) == segmentation_image, axis=0)
            masks.append(mask.astype(np.float32))
            segmentation_image[:, mask] =  0
            nonzero = np.nonzero(segmentation_image)
        if len(masks) == 0:
            raise IOError(f"Image '{fpath}' does not contain any classes.")
        segmentation = np.stack(masks, axis=0)
        return segmentation, colors

    @staticmethod
    def save_segmentation_as_image(fpath, segmentation, colors):
        """Save segmentation as color image

        Args:
            fpath: path to save the segmentation.
            segmentation: A segmentation of shape [channel, height, width].
            colors: list of colors.
        """
        # Convert to color image
        segmentation_image = np.zeros(
            (3, segmentation.shape[1], segmentation.shape[2]))
        for i, mask in enumerate(segmentation):
            segmentation_image[:, mask > 0.5] = colors[i].reshape((3, 1))
            # TODO argmax for > 2 Classes
        # Save as PIL Image
        segmentation_image = np.transpose(segmentation_image, (1, 2, 0))
        segmentation_image = (segmentation_image * 255).astype(np.uint8)
        segmentation_image = Image.fromarray(segmentation_image)
        segmentation_image.save(fpath)

class SVCDSegmentation():
    """SVCD Algorithm

    Implementation of the Algorithm "Spatially Varying Color Distributions 
    for Interactive Multilabel Segmentation" of Nieuwenhuis, Cremers.

    Attributes:
        lambda_data: Smoothness factor, regulating the strength of the dataterm.
        sigma: Color kernel variance.
        alpha: Distance factor, determining the spatial scribble influence.
        gamma: Parameter of the edge indicator function g.
        max_iter: Number of iteration of the algorithm.
        use_tqdm: Use tqdm to log progress. Default will only use tqdm if found.
    """

    def __init__(self, lambda_data=0.008, sigma=0.3, alpha=1.8, gamma=5,
                 max_iter=1000, use_tqdm: Optional[bool] = None):
        """Initializes the SVCD Algorithm."""
        self.logger = logging.getLogger(__name__)
        self.lambda_data = lambda_data
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter
        self.use_tqdm = use_tqdm

    def init_stepsizes(self):
        """Initializes the stepsizes of the Primal Dual optimization.

        Returns:
            The recommended stepsizes for the primal (first number) and the dual
                (second number) function.
        """
        return 0.5, 0.25

    def get_grid(self, shape):
        """Creates a meshgrid.

        Args:
            shape: the height and width of the image

        Returns:
            A grid of shape [2, height, width].
        """
        x = np.arange(0, shape[1], 1)
        y = np.arange(0, shape[0], 1)
        xx, yy = np.meshgrid(x, y)
        grid = np.stack((yy, xx), axis=0)
        return grid

    def init_dataterm(self, image, scribble):
        """Creates the dataterm f (eq.8,9)

        Args:
            image: the RGB image of size [channel, height, width].
            scribble: the scribbles of shape [class, height, width].

        Returns:
            the dataterm f with the dimensions [class, height, width].
        """
        c, h, w = scribble.shape
        xyScale = 1.0 / max(w, h)
        chunksize = 500
        nr_labeled_pixels = np.sum(scribble)
        # Compute distances to nearest sample point for each class
        grid = self.get_grid((h, w))[:,None,:,:].astype(np.float32)
        closest_list = []
        for i in range(c):
            nonzero = np.stack(np.nonzero(scribble[i]))[:,:,None,None]
            closest = np.ones((h,w)) * np.inf
            num_chunks = int(np.ceil(nonzero.shape[1] / chunksize))
            for chunk in range(num_chunks):
                from_idx = chunk * chunksize
                to_idx = from_idx + chunksize
                closest_other = np.min(np.sum(
                    (grid - nonzero[:,from_idx:to_idx].astype(np.float32))**2,
                    axis=0), axis=0)
                closest = np.minimum(closest, closest_other)
            closest_list.append(closest)
        rho = self.alpha * np.sqrt(np.stack(closest_list, axis=0))
        rho = np.clip(rho, 1, None)
        # Compute f
        f_list = []
        for i in range(c):
            nonzero = np.stack(np.nonzero(scribble[i]))[:,:,None,None]
            f = np.zeros((h, w))
            num_chunks = int(np.ceil(nonzero.shape[1] / chunksize))
            for chunk in range(num_chunks):
                from_idx = chunk * chunksize
                to_idx = from_idx + chunksize
                spatial_dist = np.sum(
                    (grid - nonzero[:,from_idx:to_idx].astype(np.float32))**2,
                    axis=0)
                color_dist = np.sqrt(np.sum((
                    image[:,None] - image[
                        :,
                        nonzero[0,from_idx:to_idx,0,0],
                        nonzero[1,from_idx:to_idx,0,0]
                    ][:,:,None,None])**2, axis=0))
                spatial_dist = spatial_dist * xyScale ** 2
                color_prob = (np.exp(-0.5 * np.square(color_dist/self.sigma)) /
                    (self.sigma*2.50662827463))
                spatial_prob = (np.exp(-0.5 * np.square(spatial_dist/rho[i])) /
                    (rho[i]*2.50662827463))
                f += np.sum(spatial_prob * color_prob, axis=0)
            f_list.append(f)
        P = np.stack(f_list, axis=0)
        P = P / nr_labeled_pixels
        f = - np.log(P)
        return f



    def init_halfg(self, image):
        """Creates 1/2*g(x) (eq.16)

        Measures perimeter of each set.

        Args:
            image: the RGB image with the dimensions [channel, height, width].

        Returns:
            1/2*g(x)
        """
        grayscale_img = np.mean(image, axis=0)[None]
        deriv_img = self.derivative(grayscale_img)

        deriv_img = np.sum(np.abs(deriv_img), axis=0)
        halfg = np.exp(-self.gamma * deriv_img)/2
        return halfg


    def make_derivative_matrix(self, w, h):
        """Creates a matrixform of the first order differencing.

        Args:
            w: width of the image to be differentiated
            h: height of the image to be differentiated

        Returns:
            First order differencing in matrix form.
        """
        def generate_D_matrix(n):
            e = np.ones([2,n])
            e[1,:] = -e[1,:] 
            return sparse.spdiags(e, [0,1], n, n)

        Dy = sparse.kron(sparse.eye(w), generate_D_matrix(h)) 
        Dx = sparse.kron(generate_D_matrix(w),sparse.eye(h)) 

        D = sparse.vstack([Dy, Dx])
        return D

    def derivative(self, array):
        """Creates the derivative of an input array.

        Args: 
            array: input with dimension: [class, height, width]

        Returns:
            Derivative [2, class, height, width] of the input array.
        """
        c, h, w = array.shape
        dxy = self.deriv @ np.transpose(array, (2, 1, 0)).reshape(-1,c)
        dxy = np.transpose(dxy.reshape(2, w, h, c), (0, 3, 2, 1))
        return dxy

    def divergence(self, array):
        """Creates the divergence of an input array.

        Args: 
            array: input with dimension: [2, class, height, width]

        Returns:
            Divergence [2, class, height, width] of the input array.
        """
        c, h, w = array.shape[1:]
        dxy = self.deriv.T @ np.transpose(array, (0, 3, 2, 1)).reshape(-1,c)
        dxy = np.transpose(dxy.reshape(w, h, c), (2, 1, 0))
        return -dxy

    def projection_kappa(self, xi, halfg):
        """Projection |xi_i|<=g/2, Eq.(23). 

        Args: 
            xi: input of dimension [2, class, height, width]
            halfg: 1/2 g, initialized by init_halfg(...)

        Returns:
            Projected input xi onto |xi_i|<=g/2.
        """
        norm_xi = np.sqrt(xi[0]**2 + xi[1]**2) / halfg 
        const = norm_xi>1.0
        xi[0][const] = xi[0][const] / norm_xi[const] # x
        xi[1][const] = xi[1][const] / norm_xi[const] # y
        return xi

    def projection_simplex(self, v):
        """Projection onto a simplex.
        
        As described in Algorithm 1 of
        https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
        min_w 0.5||w-v||² st sum_i w_i = z, w_i >= 0

        Args: 
            v: input array of dimension [class, height, width]

        Returns:
            Projection of the input v onto a simplex.
        """
        nc, height, width= v.shape
        # sort v into mu: mu_1 >= mu_2 >= ... mu_p
        v2d = v.reshape(nc, -1)
        mu = np.sort(v2d, axis = 0)[::-1]
        # Find p
        A = np.ones([nc,nc])
        z = 1
        sum_vecs = (np.tril(A) @ mu) - z
        c_vec = np.arange(nc)+1.
        c_vec=np.expand_dims(c_vec, axis=0).T
        cond = (mu - 1/c_vec * sum_vecs) > 0
        cond_ind = c_vec * cond
        p = np.max(cond_ind, axis=0)
        pn =np.expand_dims(p.astype(int)-1,0)
        # Calculate Theta by selecting p-entry from sum_vecs
        theta = 1 / p * np.take_along_axis(sum_vecs, indices=pn, axis=0)
        # Calculate w
        w = v2d-theta
        w[w<0] = 0
        w = w.reshape([nc,height,width])
        tmp = np.clip(v,0.000001,1)
        tmp = tmp / np.sum(tmp, axis=0, keepdims=True)
        return w

    def energy(self, theta, halfg, f):
        """Energy (eq. 17).

        Args: 
            theta: the current segmentation of dimension [class, width, height]
            halfg: 1/2 g, initialized by init_halfg(...)
            f: the dataterm create by init_dataterm(...)

        Returns:
            The energy of the current iterate.
        """
        d_xi = self.derivative(theta) # cwh --> xycwh
        norm_grad_xi = np.sqrt(d_xi[0]**2 + d_xi[1]**2)
        energy_reg = np.sum(halfg * norm_grad_xi)
        energy_dat = np.sum(theta * f)
        energy = energy_reg + energy_dat
        return energy

    def primal_energy(self, theta, halfg, f):
        """Primal Energy (eq 29).

        Args: 
            theta: the current segmentation of dimension [class, width, height]
            halfg: 1/2 g, initialized by init_halfg(...)
            f: the dataterm create by init_dataterm(...)

        Returns:
            The primal energy of the current iterate.
        """
        dtheta = self.derivative(theta)
        part1 = self.lambda_data * np.sum(theta * f)
        part2 = np.sum(halfg * np.sqrt(dtheta[0]**2 + dtheta[1]**2))
        return part1 + part2

    def dual_energy(self, xi, f):
        """Dual Energy (eq 31).

        Args: 
            xi: dual variable
            f: the dataterm create by init_dataterm(...)

        Returns:
            The dual energy of the current iterate.
        """
        return np.sum(
            np.min(self.lambda_data * f - self.divergence(xi), axis=0))


    def dual_update(self, xi_old, theta_bar, tau_dual, halfg):
        """Dual Update Step (eq.28).

        Args: 
            xi_old: dual variable
            theta_bar: "Primal Dual Hybrid Gradient" variable based on the
                       current segmentation
            tau_dual: dual stepsize
            halfg: 1/2 g, initialized by init_halfg(...)

        Returns:
            The dual update.
        """
        xi = xi_old + tau_dual * self.derivative(theta_bar)
        return self.projection_kappa(xi, halfg)

    def primal_update(self, theta_old, xi, tau_primal, f):
        """Primal Update Step (eq.28).

        Args: 
            theta_old: the current segmentation
            xi: dual update result
            tau_primal: primal stepsize
            f: the dataterm create by init_dataterm(...)

        Returns:
            The primal update.
        """
        theta = theta_old + tau_primal * (
            self.divergence(xi) - self.lambda_data * f)
        return self.projection_simplex(theta)

    def __call__(self, image, scribble):
        """Run SVCD Algorithm

        Args:
            image: input RGB image.
            scribble: predifined scribbles of shape [channel, height, width]

        Returns:
            The segmentation result and a tupel inclusing the energy values, 
            the primal energy values and the dual energy values.
        """
        # Initialize
        c, h, w = scribble.shape
        xi = np.zeros((2, c, h, w))
        theta = np.zeros((c, h, w))
        theta_bar = np.zeros((c, h, w))
        self.deriv = self.make_derivative_matrix(w,h)
        f = self.init_dataterm(image, scribble)
        halfg = self.init_halfg(image)
        tau_primal, tau_dual = self.init_stepsizes()
        energies = []
        primal_energies, dual_energies = [], []
        iterator = range(self.max_iter)
        if tqdm is not None:
            if self.use_tqdm is None or self.use_tqdm:
                iterator = tqdm(iterator)
        elif self.use_tqdm:
            raise RuntimeWarning(
                "tqdm logging requested, but tqdm could not be imported.")
        # Iterate
        for iteration in iterator:
            # Store old values
            xi_old = xi
            theta_old = theta
            # Do updates
            xi = self.dual_update(xi_old, theta_bar, tau_dual, halfg)
            theta = self.primal_update(theta_old, xi, tau_primal, f)
            theta_bar = 2 * theta - theta_old
            # Save energies
            energies.append(self.energy(theta, halfg, f))
            primal_energies.append(self.primal_energy(theta, halfg, f))
            dual_energies.append(self.dual_energy(xi, f))
        # Cleanup
        del self.deriv
        return theta, (energies, primal_energies, dual_energies)
