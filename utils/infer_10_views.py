import torch

def infer_10_views(model,img,patch_size):
    """
        Do 10 views data augmentation
        (i.e.extract 5 random patch and their horizontal flips, hence 10 patches in all)

        Args:
            model (Module): model to inter
            img (Tensor):  input tenser e.g.(b x 3 x 256 x 256)
            patch_size (int): patch size
        Returns:
            output (Tensor): output tensor
    """
    # take img size (b x 3 x 256 x 256) and patch_size = 221 as an example
    size_x = img.size()[2]  # 256
    size_y = img.size()[3]  # 256
    gap_x = size_x-patch_size
    gap_y = size_y-patch_size

    patches = [
        img[:,:,0:patch_size,0:patch_size],  # top-left
        img[:,:,gap_x:,0:patch_size],  # top-right
        img[:,:,0:patch_size,gap_y:],  # bottom-left
        img[:,:,gap_x:,gap_y:],  # bottom-right
        img[:,:,int(0.5*gap_x):int(0.5*(size_x+patch_size)),int(0.5*gap_y):int(0.5*(size_y+patch_size))]
    ]
    predictions = []
    for i,patch in enumerate(patches):  # patch size (b x 3 x 221 x 221)
        patch_flip = torch.flip(patch,[3]) # horizontal flip
        pred = torch.stack([model(patch),model(patch_flip)]).mean(0)  # (b x NUM_CLASSES)
        predictions.append(pred)
    final_pred = torch.stack(predictions).mean(0)  # (5 x b x NUM_CLASSES) -> (b x NUM_CLASSES)
    return final_pred
