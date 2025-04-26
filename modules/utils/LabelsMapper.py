class LabelsMapper:
    def __init__(self, original_array):
        self.original_array = original_array
        self.label_map_dict, self.mapped_array = self.map_array()

    def map_array(self):
        unique_values = list(set(self.original_array))
        class_mapping = {val: idx for idx, val in enumerate(unique_values)}
        classified_arr = [class_mapping[val] for val in self.original_array]
        return class_mapping, classified_arr

    def map_array_using_dict(self, array):
        mapped_array = [self.label_map_dict[val] if val in self.label_map_dict else val for val in array]
        return mapped_array
