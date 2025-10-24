import albumentations as A
from albumentations import ToTensorV2

def center_crop_square(image, **kwargs):
    """Crops the largest possible square from the center of the image."""
    h, w = image.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return image[top : top + min_dim, left:left + min_dim]

def transforms():
    """Returns list of training transformations and testing transformation"""
    # Transformation Definitions
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # 1: Normal transforms, No Aspect Ratio Preservation
    train_transforms_1 = A.Compose([
        A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p = 0.5),
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 10, val_shift_limit = 10, p = 0.5),
        A.OneOf([ 
            A.GaussianBlur(blur_limit = (3, 5), p = 1.0),
            A.Sharpen(alpha = (0.1, 0.3), lightness = (0.7, 1.0), p = 1.0),
        ], p = 0.3),
        A.Resize(224, 224),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])

    # 2: Normal transforms, Preserves Aspect Ratio
    train_transforms_2 = A.Compose([
        A.Lambda(image = center_crop_square),
        A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p = 0.5),
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 10, val_shift_limit = 10, p = 0.5),
        A.OneOf([ 
            A.GaussianBlur(blur_limit = (3, 5), p = 1.0),
            A.Sharpen(alpha = (0.1, 0.3), lightness = (0.7, 1.0), p = 1.0),
        ], p = 0.3),
        A.Resize(224, 224),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])

    # 3: Normal transforms, Preserves Aspect Ratio, Flips Horizantally
    train_transforms_3 = A.Compose([
        A.Lambda(image = center_crop_square),
        A.HorizontalFlip(p = 1),
        A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p = 0.5),
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 10, val_shift_limit = 10, p = 0.5),
        A.OneOf([ 
            A.GaussianBlur(blur_limit = (3, 5), p = 1.0),
            A.Sharpen(alpha = (0.1, 0.3), lightness = (0.7, 1.0), p = 1.0),
        ], p = 0.3),
        A.Resize(224, 224),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])

    # 4: Normal transforms, Aspect Ratio not Preserved, + or - 20 percent zoom
    train_transforms_4 = A.Compose([
        A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p = 0.5),
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 10, val_shift_limit = 10, p = 0.5),
        A.OneOf([ 
            A.GaussianBlur(blur_limit = (3, 5), p = 1.0),
            A.Sharpen(alpha = (0.1, 0.3), lightness = (0.7, 1.0), p = 1.0),
        ], p = 0.3),
        A.Affine(
            scale = (1 - 0.2, 1 + 0.2),        
            fit_output = False,
            border_mode = 0,
            p = 1
        ),
        A.Resize(224, 224),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])

    # 5: Normal transforms, Aspect Ratio Preserved, + or - 20 percent translate
    train_transforms_5 = A.Compose([
        A.Lambda(image = center_crop_square),
        A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p = 0.5),
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 10, val_shift_limit = 10, p = 0.5),
        A.OneOf([ 
            A.GaussianBlur(blur_limit = (3, 5), p = 1.0),
            A.Sharpen(alpha = (0.1, 0.3), lightness = (0.7, 1.0), p = 1.0),
        ], p = 0.3),
        A.Affine(
            translate_percent = (-0.2, 0.2),        
            fit_output = False,
            border_mode = 0,
            p = 1
        ),
        A.Resize(224, 224),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])

    # 6: Normal transforms, Aspect Ratio Preserved, + or - 20% zoom, + or - 20% translate
    train_transforms_6 = A.Compose([
        A.Lambda(image = center_crop_square),
        A.HorizontalFlip(p = 1),
        A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p = 0.5),
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 10, val_shift_limit = 10, p = 0.5),
        A.OneOf([ 
            A.GaussianBlur(blur_limit = (3, 5), p = 1.0),
            A.Sharpen(alpha = (0.1, 0.3), lightness = (0.7, 1.0), p = 1.0),
        ], p = 0.3),
        A.Affine(
            scale = (1 - 0.2, 1 + 0.2),    
            translate_percent = (-0.2, 0.2),       
            fit_output = False,
            border_mode = 0,
            p = 1
        ),
        A.Resize(224, 224),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])

    # 7: Normal transforms, Aspect Ratio Preserved, + or - 10% rotation
    train_transforms_7 = A.Compose([
        A.Lambda(image = center_crop_square),
        A.Rotate(limit = 10, p = 1),
        A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit = 0.1, p = 0.5),
        A.HueSaturationValue(hue_shift_limit = 10, sat_shift_limit = 10, val_shift_limit = 10, p = 0.5),
        A.OneOf([ 
            A.GaussianBlur(blur_limit = (3, 5), p = 1.0),
            A.Sharpen(alpha = (0.1, 0.3), lightness = (0.7, 1.0), p = 1.0),
        ], p = 0.3),
        A.Resize(224, 224),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])

    train_transforms = [train_transforms_1, train_transforms_2, train_transforms_3,
                        train_transforms_4, train_transforms_5, train_transforms_6, train_transforms_7]

    # Test transforms: Preserved Aspect Ratio and resize
    test_transform = A.Compose([
        A.Lambda(image = center_crop_square), 
        A.Resize(224, 224),
        A.Normalize(mean = mean, std = std),
        ToTensorV2()
    ])
    return train_transforms, test_transform