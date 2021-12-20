import numpy as np
import numpy.linalg as linalg

class Problem2:

    def euclidean_square_dist(self, features1, features2):
        """ Computes pairwise Euclidean square distance for all pairs.
        Args:
            features1: (128, m) numpy array, descriptors of first image
            features2: (128, n) numpy array, descriptors of second image
        Returns:
            distances: (n, m) numpy array, pairwise distances
        """
        M = np.shape(features1)[1] # M == 123
        N = np.shape(features2)[1] # N == 140
        distance = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                distance[i, j] = (linalg.norm(features2[:, i] - features1[:, j]))**2
        
        return distance

    def find_matches(self, p1, p2, distances):
        """ Find pairs of corresponding interest points given the
        distance matrix.
        Args:
            p1: (m, 2) numpy array, keypoint coordinates in first image
            p2: (n, 2) numpy array, keypoint coordinates in second image
            distances: (n, m) numpy array, pairwise distance matrix
        Returns:
            pairs: (min(n,m), 4) numpy array s.t. each row holds
                the coordinates of an interest point in p1 and p2.
        """
        M = np.shape(p1)[0] # M == 123
        N = np.shape(p2)[0] # N == 140
        pairs = np.zeros((min(N,M), 4)) # (123, 4)
        for i in range(min(N, M)):
            pairs[i, :2] = p1[i, :]
            y = np.argmin(distances[:,i])
            pairs[i, 2:4] = p2[y, :]
        
        return pairs

    def pick_samples(self, p1, p2, k):
        """ Randomly select k corresponding point pairs.
        Args:
            p1: (n, 2) numpy array, given points in first image
            p2: (m, 2) numpy array, given points in second image
            k:  number of pairs to select
        Returns:
            sample1: (k, 2) numpy array, selected k pairs in left image
            sample2: (k, 2) numpy array, selected k pairs in right image
        """

        """ hier p1 und p2 are form pairs """
        n = np.shape(p1)[0]
        sample_num = np.random.choice(n, k)  # random choose the keypoints
        sample1 = np.zeros((k, 2))
        sample2 = np.zeros((k, 2))

        """ choose the corespondence paris """
        for i in range(k):
            sample1[i, :] = p1[sample_num[i], :]
            sample2[i, :] = p2[sample_num[i], :]
        
        return sample1, sample2


    def condition_points(self, points):
        """ Conditioning: Normalization of coordinates for numeric stability 
        by substracting the mean and dividing by half of the component-wise
        maximum absolute value.
        Further, turns coordinates into homogeneous coordinates.
        Args:
            points: (l, 2) numpy array containing unnormailzed cartesian coordinates.
        Returns:
            ps: (l, 3) numpy array containing normalized points in homogeneous coordinates.
            T: (3, 3) numpy array, transformation matrix for conditioning
        """
        a = np.ones(4)
        tx, ty = np.mean(points, axis=0)
        sx, sy = np.max(abs(points), axis=0)
        T = np.array([[1/sx, 0, -tx/sx], [0, 1/sy, -ty/sy], [0, 0, 1]])
        x_homo = np.insert(points, 2, values=a, axis=1)
        
        u_T = np.dot(T, x_homo.T)
        u = u_T.T  # u has size(4x3)

        return u, T



    def compute_homography(self, p1, p2, T1, T2):
        """ Estimate homography matrix from point correspondences of conditioned coordinates.
        Both returned matrices should be normalized so that the bottom right value equals 1.
        You may use np.linalg.svd for this function.
        Args:
            p1: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img1
            p2: (l, 3) numpy array, the conditioned homogeneous coordinates of interest points in img2
            T1: (3,3) numpy array, conditioning matrix for p1
            T2: (3,3) numpy array, conditioning matrix for p2
        
        Returns:
            H: (3, 3) numpy array, homography matrix with respect to unconditioned coordinates
            HC: (3, 3) numpy array, homography matrix with respect to the conditioned coordinates
        """

        """ construct A """
        l = np.shape(p1)[0]
        A = np.zeros((2*l, 9))

        for i in range(l):
            A[2 * i, :] = [0, 0, 0, p1[i, 0], p1[i, 1], 1, -p1[i, 0] * p2[i, 1],
                           -p1[i, 1] * p2[i, 1], -p2[i, 1]]
            A[2 * i + 1, :] = [-p1[i, 0], -p1[i, 1], -1, 0, 0, 0, p1[i, 0] * p2[i, 0],
                               p1[i, 1] * p2[i, 0], p2[i, 0]]

        """ compute Homography H through SVD """
        U, S, V_T = linalg.svd(A)
        h = V_T.T[:, 8]  # the last right singular vector
        HC = h.reshape((3, 3))
        H = np.dot(np.dot(linalg.inv(T2), HC), T1)  # H = inv(T')*H_*T
        HC = HC / HC[-1, -1]  # normalize
        H = H / H[-1, -1]  # normalize

        return H, HC



    def transform_pts(self, p, H):
        """ Transform p through the homography matrix H.  
        Args:
            p: (l, 2) numpy array, interest points
            H: (3, 3) numpy array, homography matrix
        
        Returns:
            points: (l, 2) numpy array, transformed points
        """
        l = np.shape(p)[0]
        a = np.ones(l)
        pH = np.insert(p, 2, values=a, axis=1)    # homogeneous points (l, 3)
        pT = np.dot(H, pH.T)
        pt = np.array(pT.T)[:, :2]

        return pt


    def compute_homography_distance(self, H, p1, p2):
        """ Computes the pairwise symmetric homography distance.
        Args:
            H: (3, 3) numpy array, homography matrix
            p1: (l, 2) numpy array, interest points in img1
            p2: (l, 2) numpy array, interest points in img2
        
        Returns:
            dist: (l, ) numpy array containing the distances
        """
        l = np.shape(p1)[0]
        dist = np.zeros((l, 1))
        p1_trans = self.transform_pts(p1, H)
        p2_trans = self.transform_pts(p2, linalg.inv(H))
        for i in range(l):
            dist[i] = (linalg.norm(p1_trans[i, :] - p2[i, :])**2 + linalg.norm(p1[i, :] - p2_trans[i, :])**2)**0.5 #pow(0.5) == square root

        return dist


    def find_inliers(self, pairs, dist, threshold):
        """ Return and count inliers based on the homography distance. 
        Args:
            pairs: (l, 4) numpy array containing keypoint pairs
            dist: (l, ) numpy array, homography distances for k points
            threshold: inlier detection threshold
        
        Returns:
            N: number of inliers
            inliers: (N, 4)
        """
        l = np.shape(pairs)[0]
        n = 0
        inliers = []
        for i in range(l):
            if dist[i] <= threshold:
                n = n + 1
                inliers.append(pairs[i, :])

        inliers = np.array(inliers).reshape((n, 4))
        return n, inliers


    def ransac_iters(self, p, k, z):
        """ Computes the required number of iterations for RANSAC.
        Args:
            p: probability that any given correspondence is valid
            k: number of pairs
            z: total probability of success after all iterations
        
        Returns:
            minimum number of required iterations
        """
        n_iters = np.log(1 - z) / np.log(1 - p**k)
        n_iters = int(np.ceil(n_iters))  # round up and change to int type

        return n_iters


    def ransac(self, pairs, n_iters, k, threshold):
        """ RANSAC algorithm.
        Args:
            pairs: (l, 4) numpy array containing matched keypoint pairs
            n_iters: number of ransac iterations
            threshold: inlier detection threshold
        
        Returns:
            H: (3, 3) numpy array, best homography observed during RANSAC
            max_inliers: number of inliers N
            inliers: (N, 4) numpy array containing the coordinates of the inliers
        """
        """ we seperate paris at first and get p1, p2 with same shape """
        p1 = pairs[:, :2]
        p2 = pairs[:, 2:4]
        num_inliers = np.zeros(n_iters)
        H_sum = []
        inliers_sum = []
        for i in range(n_iters):

            sample1, sample2 = self.pick_samples(p1, p2, k)
            x1_homo, T1 = self.condition_points(sample1)
            x2_homo, T2 = self.condition_points(sample2)
            H, HC = self.compute_homography(x1_homo, x2_homo, T1, T2)
            H_sum.append(H)
            dist = self.compute_homography_distance(H, p1, p2)  # calculate all distances

            n, inliers = self.find_inliers(pairs, dist, threshold)  # find inliers
            num_inliers[i] = n
            inliers_sum.append(inliers)


        max_inliers = max(num_inliers)
        index_max = np.argmax(num_inliers)
        H_max_inliers = H_sum[index_max]
        inliers_max = inliers_sum[index_max]

        return H_max_inliers, max_inliers, inliers_max


    def recompute_homography(self, inliers):
        """ Recomputes the homography matrix based on all inliers.
        Args:
            inliers: (N, 4) numpy array containing coordinate pairs of the inlier points
        
        Returns:
            H: (3, 3) numpy array, recomputed homography matrix
        """
        n = np.shape(inliers)[0]
        p1 = inliers[:, :2]
        p2 = inliers[:, 2:4]
        A = np.zeros((2 * n, 9))
        for i in range(n):
            A[2 * i, :] = [0, 0, 0, p1[i, 0], p1[i, 1], 1, -p1[i, 0] * p2[i, 1],
                           -p1[i, 1] * p2[i, 1], -p2[i, 1]]
            A[2 * i + 1, :] = [-p1[i, 0], -p1[i, 1], -1, 0, 0, 0, p1[i, 0] * p2[i, 0],
                               p1[i, 1] * p2[i, 0], p2[i, 0]]

        u, s, v_T = linalg.svd(A)
        h = v_T.T[:, 8]  # the last right singular vector
        H = h.reshape((3, 3))
        H = H / H[-1, -1]  # normalize

        return H