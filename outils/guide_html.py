"""Convertit GUIDE_UTILISATEUR.md en page HTML lisible dans un navigateur.

Lancé avant la fabrication de l'application :  python outils/guide_html.py
"""
from pathlib import Path

import markdown

ICI = Path(__file__).resolve().parent.parent
texte = (ICI / "GUIDE_UTILISATEUR.md").read_text(encoding="utf-8")
corps = markdown.markdown(texte, extensions=["tables", "fenced_code"])
page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retranscription de réunions — guide</title>
<style>
 body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; max-width: 46rem;
        margin: 2rem auto; padding: 0 1rem; line-height: 1.55; color: #1d1d1f; background: #fff; }}
 h1 {{ font-size: 1.6rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }}
 code, pre {{ background: #f3f3f3; border-radius: 4px; }} code {{ padding: 0 .25rem; }}
 pre {{ padding: .6rem .8rem; overflow-x: auto; }}
 table {{ border-collapse: collapse; }} td, th {{ border: 1px solid #ccc; padding: .3rem .6rem; text-align: left; }}
 hr {{ border: 0; border-top: 1px solid #eee; margin: 2rem 0; }}
</style></head><body>
{corps}
</body></html>
"""
(ICI / "GUIDE_UTILISATEUR.html").write_text(page, encoding="utf-8")
print("GUIDE_UTILISATEUR.html écrit.")
