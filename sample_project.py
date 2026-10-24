"""Sample project "The Last Bus" (a fictional creative prompt).

Three variants:
  * sample_brief()       - brief only; run the real model pipeline on it.
  * sample_full()        - brief + a STATIC hand-written script/scenes/shots, marked provenance='sample'.
                           Lets the deterministic tools be demonstrated without any model.
  * sample_with_flaws()  - the static sample with deliberate mistakes, to show validation findings.

Static content is never presented as model output: provenance is 'sample' and exports say so.
"""
from __future__ import annotations

from schemas import (
    BudgetLevel,
    CameraMovement,
    Character,
    ProductionPlan,
    Project,
    Scene,
    ScriptOutput,
    ScriptScene,
    Shot,
)
from tools import estimate_resources

PREMISE = (
    "Бараг хоосон автобусны зогсоолд хүлээж буй залуу эмэгтэй 10 секундийн дараах ирээдүйг "
    "харж чаддагаа олж мэднэ. Нэг үзэгдэл нь түүний хажууд хүлээж буй танихгүй хүний тухай "
    "нууцыг илчилж, түүний ойлголтыг өөрчилнө. "
    "(A young woman at a nearly empty bus stop discovers she can glimpse ten seconds into the future; "
    "one vision reveals a secret about the stranger waiting beside her.)"
)


def sample_brief() -> Project:
    return Project(
        name="The Last Bus",
        genre="Нууцлаг триллер (Mystery thriller)",
        duration_minutes=3,
        max_locations=1,
        max_actors=2,
        budget=BudgetLevel.LOW,
        visual_style="Cinematic, rainy night, suspenseful atmosphere",
        constraints="Нэг байршил, хоёр жүжигчин, бага төсөв; ганц гэрэлтүүлгийн иж бүрдэл.",
        equipment=["Camera (mirrorless)", "Tripod", "Lavalier microphone", "2 LED panels", "Rain cover"],
        idea=PREMISE,
    )


_CHARACTERS = [
    Character(name="Номин", role="protagonist",
              description="Ажлаасаа оройтож харьж яваа залуу эмэгтэй; ирээдүйг 10 секундээр харна.",
              details=["улаан шүхэр", "шар борооны цув"]),
    Character(name="Танихгүй эр", role="supporting",
              description="Нам гүм, нууцлаг; Номинд ямар нэг зүйл хэлэх гэсэн мэт.",
              details=["саарал урт пальто", "доош татсан малгай"]),
]

_SCRIPT = [
    ScriptScene(number=1, heading="EXT. АВТОБУСНЫ ЗОГСООЛ - ШӨНӨ", text=(
        "Бороо зүсэрнэ. Номин улаан шүхэртэй, бараг хоосон зогсоолд ганцаараа зогсож байна. "
        "Тэр утсаа харна: 23:31. Сүүлийн автобус 23:40-д ирнэ.\n"
        "Танихгүй эр саарал урт пальтотой, малгайгаа доош татсан хэвээр орж ирээд хажууд нь суудаг.\n"
        "ТАНИХГҮЙ ЭР: (намуухан) Сүүлийн автобус үргэлж хоцордог.")),
    ScriptScene(number=2, heading="EXT. АВТОБУСНЫ ЗОГСООЛ - ШӨНӨ", text=(
        "Номин утсаа халаасалж байтал харц нь бүдгэрнэ. Дуу намжина. Тэр 10 секундийн дараахыг харна: "
        "танихгүй эр дотуур халаасаасаа Номины зургийг гаргаж, шивнэнэ: «Тэр одоо харлаа.»\n"
        "Номин цочин бодит байдал руу буцна. Эр нам гүм сууж байна.")),
    ScriptScene(number=3, heading="EXT. АВТОБУСНЫ ЗОГСООЛ - ШӨНӨ", text=(
        "Номин шүхрээ чанга атгаж эр рүү эргэнэ.\n"
        "НОМИН: Чи хэнтэй ярьсан бэ?\n"
        "ТАНИХГҮЙ ЭР: Хэнтэй ч үгүй. (түр зогсоод) Чи ч гэсэн харсан биз дээ?\n"
        "Эр бүүдгэр зургийг үзүүлнэ: зураг дээр Номины ард автобусны гэрэл.")),
    ScriptScene(number=4, heading="EXT. АВТОБУСНЫ ЗОГСООЛ - ШӨНӨ", text=(
        "Автобусны гэрэл харанхуйг эвдэн ойртоно. Эр босч Номины тохойг зөөлөн барина.\n"
        "ТАНИХГҮЙ ЭР: Бүү суу. Би өнөөдрийг гурав дахь удаа харж байна.\n"
        "Автобус хаалгаа нээнэ. Номин хаалга руу, дараа нь эр рүү харна. Моторын чимээ тасарна.")),
]

_PLACE = "Автобусны зогсоол"
_SCENES = [
    Scene(number=1, heading=_SCRIPT[0].heading, location=_PLACE, time_of_day="NIGHT",
          characters=["Номин", "Танихгүй эр"], props=["улаан шүхэр", "утас", "автобусны хуваарийн самбар"],
          action="Номин ганцаараа хүлээнэ; танихгүй эр ирж хажууд нь суудаг.", estimated_duration_sec=40),
    Scene(number=2, heading=_SCRIPT[1].heading, location=_PLACE, time_of_day="NIGHT",
          characters=["Номин", "Танихгүй эр"], props=["улаан шүхэр", "утас", "зураг"],
          action="Номин 10 секундийн ирээдүйг харж, эр түүний зургийг гаргахыг үзнэ.", estimated_duration_sec=50),
    Scene(number=3, heading=_SCRIPT[2].heading, location=_PLACE, time_of_day="NIGHT",
          characters=["Номин", "Танихгүй эр"], props=["улаан шүхэр", "зураг"],
          action="Номин эрээс асууна; эр өөрөө ч мөн харж чадна гэдгээ мэдэгдэнэ.", estimated_duration_sec=50),
    Scene(number=4, heading=_SCRIPT[3].heading, location=_PLACE, time_of_day="NIGHT",
          characters=["Номин", "Танихгүй эр"], props=["улаан шүхэр", "зураг"],
          action="Автобус ирэхэд эр Номинг суухгүй байхыг гуйна.", estimated_duration_sec=40),
]

# (scene, type, angle, movement, description, audio, seconds)
_SHOT_ROWS = [
    (1, "ESTABLISHING", "HIGH", "STATIC", "Бороонд норсон хоосон зогсоол; улаан шүхэртэй Номин зайдаа харагдана.", "Борооны чимээ, алс холын дуу.", 10),
    (1, "CLOSE_UP", "EYE_LEVEL", "STATIC", "Номин утсаа хараад 23:31 гэдгийг харна; дэлгэцийн гэрэл нүүрийг гийнгэнэ.", "Борооны чимээ.", 8),
    (1, "MEDIUM", "EYE_LEVEL", "PAN", "Танихгүй эр орж ирээд сууна; камер түүнийг дагана.", "Шалбааг дундуурх алхамны чимээ.", 12),
    (1, "TWO_SHOT", "LOW", "STATIC", "Хоёр хүн зэрэгцэн суусан; автобусны хуваарийн самбар хоёрын дунд гэрэлтнэ.", "ТАНИХГҮЙ ЭР: Сүүлийн автобус үргэлж хоцордог.", 10),
    (2, "CLOSE_UP", "EYE_LEVEL", "STATIC", "Номин утсаа халаасалж байгаад гэнэт хөдөлгөөнгүй болно; харц бүдгэрнэ.", "Дуу намждаг, чимээ бүдгэрнэ.", 12),
    (2, "EXTREME_CLOSE_UP", "EYE_LEVEL", "STATIC", "Номины нүд; хараанд нь ирээдүйн үзэгдэл туссан.", "Тоолуурын хүнд цохилт.", 8),
    (2, "INSERT", "HIGH", "STATIC", "Эр дотуур халаасаасаа Номины зургийг гаргана.", "Цаасны сэржигнэх чимээ; шивнээ «Тэр одоо харлаа.»", 15),
    (2, "MEDIUM", "DUTCH", "HANDHELD", "Номин цочин бодит байдал руу буцна; зураг ганхана.", "Дуу хүчтэй буцаж орно.", 15),
    (3, "OVER_THE_SHOULDER", "EYE_LEVEL", "STATIC", "Номин эр рүү эргэж, шүхрээ чанга атгана.", "НОМИН: Чи хэнтэй ярьсан бэ?", 12),
    (3, "CLOSE_UP", "EYE_LEVEL", "STATIC", "Эрийн нүүр малгайн сүүдэрт; түр зогсоод бүүдгэр инээмсэглэнэ.", "ТАНИХГҮЙ ЭР: Хэнтэй ч үгүй. Чи ч гэсэн харсан биз дээ?", 14),
    (3, "INSERT", "HIGH", "STATIC", "Эр бүүдгэр зургийг үзүүлнэ; зураг дээр Номины ард автобусны гэрэл.", "Борооны чимээ нэмэгдэнэ.", 12),
    (3, "CLOSE_UP", "LOW", "STATIC", "Номины нүүр: айдас ба сониуч байдал.", "Нам гүм.", 12),
    (4, "WIDE", "EYE_LEVEL", "STATIC", "Автобусны гэрэл харанхуйг эвдэж ойртоно.", "Моторын чимээ ойртоно.", 12),
    (4, "MEDIUM", "EYE_LEVEL", "HANDHELD", "Эр босч Номины тохойг барина.", "ТАНИХГҮЙ ЭР: Бүү суу. Би өнөөдрийг гурав дахь удаа харж байна.", 14),
    (4, "CLOSE_UP", "EYE_LEVEL", "STATIC", "Номин автобусны хаалга руу, дараа нь эр рүү харна.", "Хаалга нээгдэх чимээ; дуу гэнэт тасарна.", 14),
]


def _copy(items):
    """Module-level sample data must never be shared with (and mutated through) a Project."""
    return [item.model_copy(deep=True) for item in items]


def _shots() -> list[Shot]:
    return [
        Shot(number=i, scene_number=sc, shot_type=st, camera_angle=an, movement=mv,
             description=desc, audio_notes=audio, estimated_duration_sec=sec)
        for i, (sc, st, an, mv, desc, audio, sec) in enumerate(_SHOT_ROWS, start=1)
    ]


def sample_full() -> Project:
    """Static sample package (provenance 'sample'). Plan is computed by the real rule-based tool."""
    project = sample_brief()
    project.apply_script(
        ScriptOutput(
            logline="Ирээдүйг 10 секундээр харж чадах болсон залуу эмэгтэй автобусны зогсоолд хажууд нь "
                    "суугаа танихгүй хүний нууцыг олж мэднэ.",
            synopsis=("Шөнийн борооны дунд Номин сүүлийн автобусаа хүлээнэ. Танихгүй эр хажууд нь суухад "
                      "тэр гэнэт 10 секундийн ирээдүйг харна: эр түүний зургийг халаасандаа авч явдаг. "
                      "Номин эрээс асуухад эр өөрөө ч мөн ирээдүйг хардаг болохыг мэдэгдэнэ. "
                      "Автобус ирэхэд эр түүнийг суухгүй байхыг гуйна."),
            characters=_copy(_CHARACTERS), scenes=_copy(_SCRIPT)),
        origin="sample")
    project.apply_scenes(_copy(_SCENES), origin="sample")
    project.apply_shots(_shots(), origin="sample")

    result = estimate_resources(
        scenes=project.scenes, shots=project.shots, equipment=project.equipment,
        budget=project.budget.value, max_actors=project.max_actors)
    project.apply_plan(ProductionPlan.model_validate(result.data), "deterministic")
    project.approved_decisions = ["Танихгүй эр Номиныг хамгаалахыг оролдож байгаа нь төгсгөлд илрэнэ."]
    return project


def sample_with_flaws() -> Project:
    """Static sample with deliberate mistakes so the validation tab has something to show."""
    project = sample_brief()
    scenes = _copy(_SCENES)
    scenes[2].characters = ["Номин", "Танихгүй эр", "Жолооч"]   # unknown + 3rd actor
    scenes[2].location = "Дэлгүүрийн үүдэнд"                    # 2nd location
    scenes[3].estimated_duration_sec = 95                       # total no longer ~180 s
    shots = _shots()
    shots[-1].scene_number = 9                                  # points to a missing scene
    shots[7].movement = CameraMovement.DOLLY                               # no dolly in the equipment list

    project.apply_script(
        ScriptOutput(logline="[Алдаатай жишээ] The Last Bus — validation харуулах зориулалттай.",
                     synopsis="Зориудаар алдаа оруулсан static жишээ.",
                     characters=_copy(_CHARACTERS), scenes=_copy(_SCRIPT)),
        origin="sample")
    project.apply_scenes(scenes, origin="sample")
    project.apply_shots(shots, origin="sample")
    return project

