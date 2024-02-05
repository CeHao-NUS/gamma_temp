

NEW_TARGET_GAPARTS = ['line_fixed_handle', 'round_fixed_handle', 'hinge_door', 'slider_drawer', 'slider_lid', 'hinge_lid',
                      'hinge_handle', "revolute_switch", "revolute_seat", "blade_revolute_handle", "revolute_leg"]


TARGET_GAPARTS = ['line_fixed_handle', 'round_fixed_handle', 'slider_button', 'hinge_door', 'slider_drawer','slider_lid', 'hinge_lid',
                  'hinge_knob', 'hinge_handle', "revolute_switch", "revolute_seat", "blade_revolute_handle", "revolute_leg"]

NEW_TARGET_GAPART_IDS = [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12]

NEW_TARGET_GAPART_SEM_IDS = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13]

NEW_TARGET_GAPART_SEM_IDS_TO_NEW_ids = {1:1, 2:2, 4:3, 5:4, 6:5, 7:6, 9:7, 10:8, 11:9, 12:10, 13:11}


OBJECT_CATEGORIES = [
    'Box', 'Camera', 'CoffeeMachine', 'Dishwasher', 'KitchenPot', 'Microwave', 'Oven', 'Phone', 'Refrigerator',
    'Remote', 'Safe', 'StorageFurniture', 'Table', 'Toaster', 'TrashCan', 'WashingMachine', 'Keyboard', 'Laptop', 'Door', 'Printer',
    'Suitcase', 'Bucket', 'Toilet'
]

OBJECT_NAME2ID = {
    # seen category
    "Box": 0,
    "Remote": 1,
    "Microwave": 2,
    "Camera": 3,
    "Dishwasher": 4,
    "WashingMachine": 5,
    "CoffeeMachine": 6,
    "Toaster": 7,
    "StorageFurniture": 8,
    "AKBBucket": 9, # akb48
    "AKBBox": 10, # akb48
    "AKBDrawer": 11, # akb48
    "AKBTrashCan": 12, # akb48
    "Bucket": 13, # new
    "Keyboard": 14, # new
    "Printer": 15, # new
    "Toilet": 16, # new
    # unseen category
    "KitchenPot": 17,
    "Safe": 18,
    "Oven": 19,
    "Phone": 20,
    "Refrigerator": 21,
    "Table": 22,
    "TrashCan": 23,
    "Door": 24,
    "Laptop": 25,
    "Suitcase": 26, # new
}

TARGET_GAPARTS_DICT = {"fixed_base": -1,
                       "line_fixed_handle": 0,
                       "round_fixed_handle": 1,
                       "slider_button": 2,
                       "hinge_door": 3,
                       "slider_drawer": 4,
                       "slider_lid": 5,
                       "hinge_lid": 6,
                       "hinge_knob": 7,
                       "hinge_handle": 8
                       }
# handel, drawer, 0, 1, 3, 4, 6,
# 0 is line_fixed_handle, 1 is round_fixed_handle, 3 is hinge_door, 4 is slider_drawer, 6 is hinge_lid


