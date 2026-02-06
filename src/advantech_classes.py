#!/usr/bin/env python3
"""Advantech YOLO Classes - ImageNet and COCO class definitions."""

__title__ = "Advantech YOLO Class Definitions"
__author__ = "Samir Singh"
__copyright__ = "Copyright (c) 2024-2025 Advantech Corporation. All Rights Reserved."
__version__ = "1.0.0"
__build_date__ = "2025-12-05"


import threading
from typing import List, Tuple, Optional  # 添加必要的导入

# ==========================================================================
# COCO Classes (80 classes for detection/segmentation)
# ==========================================================================

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# ==========================================================================
# ImageNet Classes (1000 classes for classification)
# ==========================================================================

IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch",
    "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay",
    "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle",
    "vulture", "great grey owl", "European fire salamander", "common newt", "eft",
    "spotted salamander", "axolotl", "bullfrog", "tree frog", "tailed frog",
    "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "common iguana", "American chameleon", "whiptail lizard",
    "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "green lizard",
    "African chameleon", "Komodo dragon", "African crocodile", "American alligator",
    "triceratops", "thunder snake", "ringneck snake", "hognose snake", "green snake",
    "king snake", "garter snake", "water snake", "vine snake", "night snake",
    "boa constrictor", "rock python", "Indian cobra", "green mamba", "sea snake",
    "horned viper", "diamondback rattlesnake", "sidewinder rattlesnake", "trilobite",
    "harvestman", "scorpion", "black and gold garden spider", "barn spider",
    "garden spider", "black widow", "tarantula", "wolf spider", "tick", "centipede",
    "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peacock",
    "quail", "partridge", "African grey parrot", "macaw", "sulphur-crested cockatoo",
    "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar",
    "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker",
    "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone",
    "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug",
    "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab",
    "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule",
    "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank",
    "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale",
    "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese",
    "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier",
    "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound",
    "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki",
    "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer",
    "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog",
    "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel",
    "Cocker Spaniels", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke",
    "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor",
    "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie",
    "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff",
    "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky",
    "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji",
    "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed",
    "Pomeranian", "Chow Chow", "Keeshond", "Brussels Griffon", "Pembroke Welsh Corgi",
    "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle",
    "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog",
    "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat",
    "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx",
    "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear",
    "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat",
    "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada",
    "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly",
    "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly",
    "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit",
    "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot",
    "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar",
    "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)",
    "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle",
    "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret",
    "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey",
    "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey",
    "marmoset", "white-headed capuchin", "howler monkey", "titi monkey",
    "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur",
    "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda",
    "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya",
    "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner",
    "airship", "altar", "ambulance", "amphibious vehicle", "analog clock",
    "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam",
    "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail",
    "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap",
    "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie",
    "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie",
    "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket",
    "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab",
    "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror",
    "carousel", "tool kit", "carton", "car wheel", "automated teller machine",
    "cassette", "cassette player", "castle", "catamaran", "CD player", "cello",
    "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs",
    "cocktail shaker", "coffee mug", "coffeepot", "coil", "combination lock",
    "computer keyboard", "confectionery", "container ship", "convertible",
    "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane",
    "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch",
    "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper",
    "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher",
    "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine",
    "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck",
    "fire screen", "flagpole", "flute", "folding chair", "football helmet",
    "forklift", "fountain", "fountain pen", "four-poster bed", "freight car",
    "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator",
    "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store",
    "guillotine", "barrette", "hair spray", "half-track", "hammer", "hamper",
    "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica",
    "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb",
    "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle",
    "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad",
    "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower",
    "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine",
    "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker",
    "loupe magnifying glass", "sawmill", "magnetic compass", "mail bag", "mailbox",
    "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba",
    "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet",
    "megalith", "microphone", "microwave oven", "military uniform", "milk can",
    "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl",
    "mobile home", "ford model t", "modem", "monastery", "monitor", "moped",
    "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van",
    "muzzle", "nail", "neck brace", "necklace", "baby pacifier", "notebook computer",
    "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope",
    "overskirt", "bullock cart", "oxygen mask", "packet", "paddle", "paddle wheel",
    "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel",
    "parachute", "parallel bars", "park bench", "parking meter", "passenger car",
    "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume",
    "Petri dish", "photocopier", "picket fence", "pickup truck", "pier",
    "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel",
    "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole",
    "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel",
    "power drill", "prayer rug", "printer", "prison", "missile", "projector",
    "hockey puck", "punching bag", "purse", "quill", "quilt", "race car",
    "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera",
    "refrigerator", "remote control", "restaurant", "revolver", "rifle",
    "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick",
    "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong",
    "saxophone", "scabbard", "weighing scale", "school bus", "schooner",
    "scoreboard", "CRT screen", "screw", "screwdriver", "seat belt", "sewing machine",
    "shield", "shoe store", "shoji screen / room divider", "shopping basket",
    "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask",
    "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel",
    "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector",
    "sombrero", "soup bowl", "space bar", "space heater", "space shuttle", "spatula",
    "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage",
    "steam locomotive", "through arch bridge", "steel drum", "stethoscope",
    "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher",
    "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses",
    "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts",
    "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player",
    "teapot", "teddy bear", "television", "tennis ball", "thatched roof",
    "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck",
    "toy store", "tractor", "semi-trailer truck", "tray", "trench coat",
    "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone",
    "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle",
    "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin",
    "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft",
    "sink", "washing machine", "water bottle", "water jug", "water tower",
    "whiskey jug", "whistle", "hair wig", "window screen", "window shade",
    "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon",
    "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket",
    "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream",
    "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog",
    "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash",
    "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper",
    "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)",
    "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf",
    "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog",
    "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory",
    "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom",
    "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn",
    "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric",
    "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom",
    "bolete", "corn cob", "toilet paper"
]

# ==========================================================================
# Helper Functions
# ==========================================================================

def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Generate a color for a class ID."""
    import random
    random.seed(class_id * 12345)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color

def get_class_name(class_id: int, task: str = "detection") -> str:
    """Get class name for a given class ID and task type."""
    if task == "classification":
        classes = IMAGENET_CLASSES
    else:
        classes = COCO_CLASSES
    
    # 检查ID是否在有效范围内
    if class_id < 0:
        return f"invalid_class_{class_id}"
    
    if class_id < len(classes):
        class_name = classes[class_id]
        # 如果类别名称为空或无效，返回unknown
        if not class_name or class_name.strip() == "":
            return f"unknown_class_{class_id}"
        return class_name
    else:
        # 对于超出范围的ID，提供更友好的信息
        if task == "classification":
            if class_id >= 1000 and class_id < 2000:
                # 可能是ImageNet 21k的类别
                return f"imagenet21k_class_{class_id}"
            else:
                return f"unknown_class_{class_id}"
        else:
            return f"class_{class_id}"

def get_class_names(task: str = "detection") -> List[str]:
    """Get the entire class list for a given task type."""
    return IMAGENET_CLASSES if task == "classification" else COCO_CLASSES

# ==========================================================================
# Custom Class Loader
# ==========================================================================

def load_custom_classes(file_path: str, task: str = "detection") -> List[str]:
    """
    Load custom classes from a text file.
    
    Args:
        file_path: Path to the text file containing class names (one per line)
        task: Task type ('detection' or 'classification')
    
    Returns:
        List of class names
    """
    try:
        with open(file_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(classes)} custom classes from {file_path}")
        return classes
    except Exception as e:
        print(f"Failed to load custom classes from {file_path}: {e}")
        # 返回默认类别
        return get_class_names(task)

# ==========================================================================
# Class Manager (Singleton)
# ==========================================================================

class AdvantechClassManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._detection_classes = COCO_CLASSES
        self._classification_classes = IMAGENET_CLASSES
        self._custom_classes = {}
        self._initialized = True
    
    def set_custom_classes(self, task: str, classes: List[str]):
        """Set custom classes for a specific task."""
        if task == "detection":
            self._detection_classes = classes
        elif task == "classification":
            self._classification_classes = classes
        else:
            # 这里不要导入AdvantechTaskType，直接使用字符串
            raise ValueError(f"Unknown task type: {task}")
    
    def get_classes(self, task: str) -> List[str]:
        """Get classes for a specific task."""
        if task == "classification":
            return self._classification_classes
        else:
            return self._detection_classes
    
    def get_class_name(self, class_id: int, task: str = "detection") -> str:
        """Get class name for a given class ID and task type."""
        classes = self.get_classes(task)
        
        if class_id < 0:
            return f"invalid_class_{class_id}"
        
        if class_id < len(classes):
            class_name = classes[class_id]
            if not class_name or class_name.strip() == "":
                return f"unknown_class_{class_id}"
            return class_name
        else:
            if task == "classification" and class_id >= 1000 and class_id < 2000:
                return f"imagenet21k_class_{class_id}"
            return f"unknown_class_{class_id}"

# Global instance
CLASS_MANAGER = AdvantechClassManager()

if __name__ == "__main__":
    # Test the class manager
    print("=" * 50)
    print("Advantech Class Definitions Test")
    print("=" * 50)
    
    # Test detection classes
    detection_classes = get_class_names("detection")
    print(f"Detection classes: {len(detection_classes)}")
    print(f"First 5: {detection_classes[:5]}")
    print(f"Last 5: {detection_classes[-5:]}")
    
    # Test classification classes
    classification_classes = get_class_names("classification")
    print(f"\nClassification classes: {len(classification_classes)}")
    print(f"First 5: {classification_classes[:5]}")
    print(f"Last 5: {classification_classes[-5:]}")
    
    # Test class name function
    print(f"\nClass name tests:")
    print(f"  Detection class 0: {get_class_name(0, 'detection')}")
    print(f"  Detection class 79: {get_class_name(79, 'detection')}")
    print(f"  Classification class 0: {get_class_name(0, 'classification')}")
    print(f"  Classification class 999: {get_class_name(999, 'classification')}")
    print(f"  Out of range class 1000: {get_class_name(1000, 'classification')}")
    
    print("\nTest completed successfully!")
