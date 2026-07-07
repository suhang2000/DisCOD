"""Prompts for VLM box generation."""

DETECT_SYSTEM = (
    "You are a camouflaged object detector. There IS a camouflaged object in this image. "
    "Output ONLY its bounding box as <bbox>[x1,y1,x2,y2]</bbox> with coordinates normalized "
    "to [0,1000] where 1000 = full image dimension. Do not output any reasoning or extra text."
)
DETECT_USER = "Output the bounding box of the camouflaged object as <bbox>[x1,y1,x2,y2]</bbox>."

DISAGREEMENT_PROMPTS = [
    ("You are a camouflaged object detector. There IS a camouflaged object. Output ONLY its bounding box as "
     "<bbox>[x1,y1,x2,y2]</bbox> normalized to [0,1000]. No reasoning.",
     "Output the camouflaged object's bounding box as <bbox>[x1,y1,x2,y2]</bbox>."),
    ("You analyze by texture. Find the region where the texture or color pattern subtly BREAKS from the "
     "surrounding background - that discontinuity is the hidden object. Output ONLY <bbox>[x1,y1,x2,y2]</bbox> in [0,1000]. No reasoning.",
     "Where does the texture break? Output that region's <bbox>[x1,y1,x2,y2]</bbox>."),
    ("You detect shapes. Look for the OUTLINE or silhouette of a living creature concealed in the scene. "
     "Output ONLY the creature's <bbox>[x1,y1,x2,y2]</bbox> in [0,1000]. No reasoning.",
     "Output the bounding box of the creature's silhouette as <bbox>[x1,y1,x2,y2]</bbox>."),
    ("You find anomalies. Identify the single region that does NOT naturally belong to the background "
     "environment. Output ONLY its <bbox>[x1,y1,x2,y2]</bbox> in [0,1000]. No reasoning.",
     "Output the bounding box of the out-of-place region as <bbox>[x1,y1,x2,y2]</bbox>."),
    ("You reason part-to-whole. Find any visible body part (eye, leg, fin, antenna, tail) of a hidden animal, "
     "then infer the WHOLE animal's bounding box as <bbox>[x1,y1,x2,y2]</bbox> in [0,1000]. No reasoning text.",
     "Find a body part, then output the whole animal's <bbox>[x1,y1,x2,y2]</bbox>."),
]
