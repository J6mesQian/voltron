import sys
sys.path.append('/home/jamesqian/knowledge_aug_proj/voltron/voltron_evaluation')

import torch
import torchvision.transforms as T

from voltron import instantiate_extractor, load
import voltron_evaluation as vet

from transformers import FlavaProcessor, FlavaForPreTraining, FlavaConfig

#load model from checkpoint
model_path = "/home/jamesqian/knowledge_aug_proj/knowledge_aug_vp/output/flava_pretraining_Ego4D/checkpoint-451197"
model = FlavaForPreTraining.from_pretrained(model_path)
#model = FlavaForPreTraining.from_pretrained('facebook/flava-full')

processor = FlavaProcessor.from_pretrained('facebook/flava-full')

processed = processor(
            images = torch.rand(3, 224, 224), 
            text = "picking something up", 
            padding="max_length", 
            return_tensors="pt",
            max_length=512,
            truncation=True,
            return_codebook_pixels=True,
            return_image_mask=False,
        )
#import ipdb; ipdb.set_trace()
NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
preprocess = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Lambda(lambda x: x * 255.0),
        ]
    )

#outputs = model(**processed)

def evaluate_grasp() -> None:
    # Load Backbone (V-Cond)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = model.to(device)
    
    #freeze backbone and set to eval mode
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    # Create MAP Extractor Factory (for ARC Grasping) --> 4 PuP stages with an 80 x 80 output resolution.
    #   => Because we're segmenting, the `n_latents` is intermingled with the details of the adapter; not really a clean
    #      way to deal with this, so we'll hardcode based on the above.
    output_resolution, upsample_stages = 80, 4
    map_extractor_fn = instantiate_extractor(backbone, n_latents=int((output_resolution**2) / (4**upsample_stages)))
    
    # Create ARC Grasping Harness
    grasp_evaluator = vet.GraspAffordanceHarness("FLAVA_Ego4D_debug", backbone, preprocess, map_extractor_fn)
    grasp_evaluator.fit()
    grasp_evaluator.test()


if __name__ == "__main__":
    evaluate_grasp()