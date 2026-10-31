"""Download the Genius Invokation tables we use from Dimbreath's AnimeGameData into data/raw/."""

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://gitlab.com/Dimbreath/animegamedata2/-/raw/main/"  # the original AnimeGameData repo is frozen at 6.6
FILES = [
    # Cards, characters and skills
    "ExcelBinOutput/GCGCardExcelConfigData.json",
    "ExcelBinOutput/GCGCharExcelConfigData.json",
    "ExcelBinOutput/GCGSkillExcelConfigData.json",
    "ExcelBinOutput/GCGTagExcelConfigData.json",
    # NPC decks and matchups
    "ExcelBinOutput/GCGDeckExcelConfigData.json",
    "ExcelBinOutput/GCGGameExcelConfigData.json",
    # Collection and progression
    "ExcelBinOutput/GCGDeckCardExcelConfigData.json",
    "ExcelBinOutput/GCGProficiencyRewardExcelConfigData.json",
    "ExcelBinOutput/GCGLevelExcelConfigData.json",
    "ExcelBinOutput/GCGWeekLevelExcelConfigData.json",
    "ExcelBinOutput/GCGChallengeExcelConfigData.json",
    "ExcelBinOutput/GCGGameRewardExcelConfigData.json",
    "ExcelBinOutput/GCGKeywordExcelConfigData.json",
    "ExcelBinOutput/GCGLevelLockExcelConfigData.json",
    "ExcelBinOutput/GCGWorldLevelExcelConfigData.json",
    "ExcelBinOutput/GCGBossLevelExcelConfigData.json",
    "ExcelBinOutput/GCGQuestLevelExcelConfigData.json",
    "ExcelBinOutput/GCGDeckStorageExcelConfigData.json",
    "ExcelBinOutput/GCGCharacterLevelExcelConfigData.json",
    "ExcelBinOutput/AvatarExcelConfigData.json",
    "ExcelBinOutput/RewardExcelConfigData.json",
    "ExcelBinOutput/MaterialExcelConfigData.json",
    "ExcelBinOutput/ShopGoodsExcelConfigData.json",
    # English names and descriptions
    "TextMap/TextMapEN.json",
    "TextMap/TextMap_MediumEN.json",
]
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for remote_path in FILES:
        destination = OUTPUT_DIR / Path(remote_path).name
        print(f"fetching {remote_path}", file=sys.stderr)
        urllib.request.urlretrieve(BASE_URL + remote_path, destination)


if __name__ == "__main__":
    main()
