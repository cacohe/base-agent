from pathlib import Path
from typing import List, Dict, Any

from src.config import settings
from src.utils.logger import logger


class SkillManager:
    def __init__(self):
        self.skills_dir = Path(settings.skills_dir)
        logger.info(f"SkillManager initialized with skills_dir: {self.skills_dir}")

    async def get_all_skills(self) -> List[Dict[str, Any]]:
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory does not exist: {self.skills_dir}")
            return []
        return [str(d) for d in self.skills_dir.iterdir() if d.is_dir()]
