from torchreid.utils import FeatureExtractor

extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='a/b/c/model.pth.tar',
    device='cuda'
)

image_list = [
    '/mnt/data_ubuntu/phongnn/Market-1501-v15.09.15/query/0001_c1s1_001051_00.jpg',
    '/mnt/data_ubuntu/phongnn/Market-1501-v15.09.15/query/0003_c1s6_015971_00.jpg',
]

features = extractor(image_list)
print(features.shape) # output (5, 512)