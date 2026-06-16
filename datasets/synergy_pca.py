import numpy as np

class PCASynergy:
    """
    Standard PCA for dimensionality reduction.
    [MODIFIED]: Removed 'whitening'. Now coefficients represent raw projections 
    (physical meaning preserved).
    """
    def __init__(self, n_components):
        self.n_components = n_components
        # 彻底移除 whiten 参数
        
        self.mean = None
        self.components = None
        self.orig_dim = None

    def fit(self, data):
        if data.ndim == 1: data = data.reshape(1, -1)
        N_samples, self.orig_dim = data.shape
        n_comp = min(self.n_components, self.orig_dim)
        self.n_components = n_comp
        
        # 1. Compute Mean & Center
        self.mean = np.mean(data, axis=0)
        data_centered = data - self.mean
        
        # 2. SVD
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        self.components = Vt[:n_comp]
        
        print(f"[PCASynergy] Fitted: {n_comp} components (No Whitening).")

    def transform(self, dof_vec):
        """ Joint Angles -> Raw Synergy Coefficients """
        if self.mean is None: raise RuntimeError("Model not fitted.")
        
        data = np.array(dof_vec)
        input_is_1d = (data.ndim == 1)
        if input_is_1d: data = data.reshape(1, -1)
            
        centered = data - self.mean
        synergy = np.dot(centered, self.components.T)
        
        # [REMOVED] No division by std
        
        if input_is_1d: return synergy.flatten()
        return synergy

    def inverse_transform(self, synergy_vec):
        """ Raw Synergy Coefficients -> Joint Angles """
        if self.mean is None: raise RuntimeError("Model not fitted.")

        data = np.array(synergy_vec)
        input_is_1d = (data.ndim == 1)
        if input_is_1d: data = data.reshape(1, -1)

        # [REMOVED] No multiplication by std
        
        reconstructed = np.dot(data, self.components)
        reconstructed = reconstructed + self.mean
        
        if input_is_1d: return reconstructed.flatten()
        return reconstructed

    def get_save_dict(self):
        return {
            "mean": self.mean,
            "components": self.components,
            "n_components": self.n_components
            # std 和 whiten 不再需要保存
        }
    
    def load_from_dict(self, save_dict):
        self.mean = save_dict["mean"]
        try:
            self.components = save_dict["components"]
        except:
            self.components = save_dict["components_"]
        try:
            self.n_components = save_dict["n_components"]
        except:
            self.n_components = self.components.shape[0]