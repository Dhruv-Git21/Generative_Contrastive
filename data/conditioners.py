import numpy as np

class LabelConditioner:
    """
    Provides a way to get condition embedding for a given label.
    Can compute class centroid embeddings from dataset.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Precompute centroid for each label
        self.label_to_centroid = {}
        if dataset.labels:
            labels = dataset.labels
            embeddings = dataset.embeddings
            label_groups = {}
            for emb, lbl in zip(embeddings, labels):
                if lbl not in label_groups:
                    label_groups[lbl] = []
                label_groups[lbl].append(np.array(emb, dtype=float))
            for lbl, emb_list in label_groups.items():
                arr = np.stack(emb_list, axis=0)
                centroid = arr.mean(axis=0)
                self.label_to_centroid[lbl] = centroid.tolist()
    
    def get_condition(self, label):
        """
        Return the embedding vector to use as condition for the given label.
        If the label exists in precomputed centroids, returns that.
        Otherwise, returns None.
        """
        return self.label_to_centroid.get(label, None)
