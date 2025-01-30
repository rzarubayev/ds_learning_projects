from pydantic import BaseModel, Field

class FlatsPriceParams(BaseModel):
    id: int
    rooms: int
    total_area: float
    kitchen_area: float
    living_area: float
    floor: int
    studio: bool
    is_apartment: bool
    building_type_int: int
    build_year: int
    latitude: float
    longitude:  float
    ceiling_height: float
    flats_count: int
    floors_total: int
    has_elevator: bool