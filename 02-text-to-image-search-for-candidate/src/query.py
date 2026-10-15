"""Conservative lexical object cues; no image IDs, benchmark queries or answers.

Unrecognized descriptions retain whole-sentence CLIP retrieval. The vocabulary
is an application-owned synonym list, not an imported dataset annotation file.
"""

import re
from dataclasses import dataclass

OBJECTS = {
    "person": ("person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children"),
    "cat": ("cat", "cats", "kitten", "kittens"),
    "dog": ("dog", "dogs", "puppy", "puppies"),
    "horse": ("horse", "horses", "pony", "ponies"),
    "elephant": ("elephant", "elephants"),
    "bird": ("bird", "birds", "pigeon", "pigeons", "seagull", "seagulls"),
    "bear": ("bear", "bears"),
    "sheep": ("sheep", "lamb", "lambs"),
    "cow": ("cow", "cows", "cattle"),
    "bicycle": ("bicycle", "bicycles", "bike", "bikes", "cycling", "cyclist", "cyclists"),
    "motorcycle": ("motorcycle", "motorcycles", "motorbike", "motorbikes", "scooter", "scooters"),
    "car": ("car", "cars", "automobile", "automobiles", "suv", "suvs"),
    "bus": ("bus", "buses"),
    "train": ("train", "trains", "locomotive", "locomotives", "tram", "trams"),
    "truck": ("truck", "trucks", "lorry", "lorries"),
    "airplane": ("airplane", "airplanes", "aeroplane", "aeroplanes", "aircraft", "plane", "planes"),
    "boat": ("boat", "boats", "ship", "ships", "sailboat", "sailboats"),
    "kite": ("kite", "kites"),
    "frisbee": ("frisbee", "frisbees", "flying disc"),
    "surfboard": ("surfboard", "surfboards", "surfer", "surfers", "surfing"),
    "skateboard": ("skateboard", "skateboards", "skateboarder", "skateboarding"),
    "umbrella": ("umbrella", "umbrellas"),
    "banana": ("banana", "bananas"),
    "apple": ("apple", "apples"),
    "orange": ("orange fruit", "oranges"),
    "pizza": ("pizza", "pizzas"),
    "sandwich": ("sandwich", "sandwiches"),
    "food": ("food", "meal", "meals", "breakfast", "lunch", "dinner"),
    "cup": ("cup", "cups", "mug", "mugs"),
    "bottle": ("bottle", "bottles"),
    "plate": ("plate", "plates"),
    "bowl": ("bowl", "bowls"),
    "chair": ("chair", "chairs"),
    "sofa": ("sofa", "sofas", "couch", "couches"),
    "bed": ("bed", "beds"),
    "bench": ("bench", "benches"),
    "table": ("table", "tables", "desk", "desks"),
    "traffic light": ("traffic light", "traffic lights"),
    "stop sign": ("stop sign", "stop signs"),
    "fire hydrant": ("fire hydrant", "fire hydrants", "hydrant", "hydrants"),
    "parking meter": ("parking meter", "parking meters"),
    "teddy bear": ("teddy bear", "teddy bears", "stuffed toy", "stuffed toys"),
    "tennis racket": ("tennis racket", "tennis rackets", "tennis racquet"),
    "knife": ("knife", "knives"),
    "fork": ("fork", "forks"),
    "spoon": ("spoon", "spoons"),
    "laptop": ("laptop", "laptops"),
    "phone": ("phone", "phones", "telephone", "telephones"),
    "clock": ("clock", "clocks"),
    "toilet": ("toilet", "toilets"),
    "refrigerator": ("refrigerator", "fridge"),
    "wheelchair": ("wheelchair", "wheelchairs"),
}

CONFUSIONS = {
    "cat": ("dog", "teddy bear"), "dog": ("cat", "teddy bear", "sheep"),
    "horse": ("cow", "elephant", "dog"), "elephant": ("horse", "cow"),
    "bird": ("airplane", "kite", "teddy bear"),
    "bicycle": ("motorcycle", "skateboard", "wheelchair"),
    "motorcycle": ("bicycle", "car"),
    "car": ("bus", "train", "truck", "motorcycle", "parking meter"),
    "bus": ("car", "train", "truck"), "train": ("bus", "truck", "car"),
    "airplane": ("bird", "kite"), "umbrella": ("kite",),
    "banana": ("apple", "orange", "frisbee"),
    "traffic light": ("stop sign", "fire hydrant", "parking meter"),
    "chair": ("sofa", "bench", "table"),
    "cup": ("bowl", "bottle"), "plate": ("bowl",),
}


@dataclass(frozen=True)
class QueryPlan:
    objects: tuple[str, ...]
    primary: str | None
    negated: bool
    ambiguous: bool = False


def parse_query(query: str) -> QueryPlan:
    text = query.lower()
    # Positive object cues are not safe for negation or alternatives. Fall back.
    negated = bool(re.search(r"\b(no|not|without|neither|nor|except|excluding|or)\b", text))
    # These noun compounds must not be interpreted as a literal animal/vehicle.
    ambiguous = bool(re.search(
        r"\b(hot dogs?|dog food|cat food|bird food|bus stops?|train stations?|car parks?|bird'?s?[- ]eye)\b", text
    ))
    matches = []
    for canonical, synonyms in OBJECTS.items():
        for synonym in synonyms:
            match = re.search(r"\b" + re.escape(synonym) + r"\b", text)
            if match:
                matches.append((match.start(), -len(synonym), canonical))
                break
    matches.sort()
    # Do not treat the word 'bear' inside 'teddy bear' as a second object.
    names = tuple(dict.fromkeys(canonical for _, _, canonical in matches))
    if "teddy bear" in names:
        names = tuple(name for name in names if name != "bear")
    candidates = [name for name in names if name != "person"]
    return QueryPlan(names, candidates[0] if candidates else ("person" if "person" in names else None), negated, ambiguous)


def object_prompt(name: str) -> str:
    article = "an" if name[0] in "aeiou" else "a"
    return f"a photo of {article} {name}."
