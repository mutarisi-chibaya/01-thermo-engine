"""Record a narrated demo video of the pyrothermo Streamlit app.

Run from 01-thermo-engine:

    .venv\\Scripts\\python.exe demo/make_demo.py

Needs: pip install edge-tts playwright ; playwright install chromium ; ffmpeg on PATH.
Narration uses the Microsoft Edge online TTS voice en-ZA-LukeNeural (the script text is
sent to Microsoft's service). Output: demo/output/pyrothermo_demo.mp4
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

import edge_tts
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "demo" / "build"
OUT = ROOT / "demo" / "output" / "pyrothermo_demo.mp4"
VOICE = "en-ZA-LukeNeural"
RATE = "-4%"
PORT = 8599
URL = f"http://localhost:{PORT}"
VIEW = {"width": 1920, "height": 1080}
GAP_S = 0.7  # pause after each narration line


# ---------------------------------------------------------------- browser actions


def _pause(page: Page, ms: int) -> None:
    page.wait_for_timeout(ms)


def _hover_chart(
    page: Page, index: int = 0, steps: int = 40, frac: tuple[float, float] = (0.1, 0.9)
) -> None:
    plot = page.locator(".js-plotly-plot:visible").nth(index)
    plot.scroll_into_view_if_needed()
    box = plot.bounding_box()
    if box is None:
        return
    y = box["y"] + box["height"] * 0.5
    for i in range(steps + 1):
        x = box["x"] + box["width"] * (frac[0] + (frac[1] - frac[0]) * i / steps)
        page.mouse.move(x, y)
        _pause(page, 70)


def _scroll_to(page: Page, text: str) -> None:
    page.get_by_text(text, exact=True).first.scroll_into_view_if_needed()


def _scroll_top(page: Page) -> None:
    page.get_by_role("heading", name="pyrothermo").first.scroll_into_view_if_needed()


def _choose_preset(page: Page, label_start: str) -> None:
    _scroll_top(page)
    page.locator('[data-testid="stSelectbox"]:visible').first.click()
    page.get_by_role("option").filter(has_text=label_start).first.click()
    _pause(page, 900)


def _set_temperature(page: Page, label: str, value: str) -> None:
    box = page.get_by_role("spinbutton", name=label)
    box.scroll_into_view_if_needed()
    box.fill(value)
    box.press("Enter")
    _pause(page, 900)


def _to_top(page: Page, heading: str) -> None:
    """Scroll so ``heading`` sits at the top of the viewport (whole section visible)."""
    page.get_by_role("heading", name=heading).first.evaluate(
        "e => e.scrollIntoView({block: 'start'})"
    )
    page.mouse.wheel(0, -40)
    _pause(page, 300)


def _show_plain(page: Page) -> None:
    _to_top(page, "In plain language")


def act_intro(page: Page) -> None:
    _pause(page, 800)


def act_tutorial(page: Page) -> None:
    _scroll_to(page, "Try it in three steps")
    _pause(page, 3500)
    page.get_by_text("Glossary of terms").click()
    _pause(page, 3000)
    page.get_by_text("Glossary of terms").click()
    _scroll_top(page)


def act_hydrogen(page: Page) -> None:
    page.get_by_role("tab", name="Reactions").click()
    _pause(page, 1500)
    _show_plain(page)


def act_specialist(page: Page) -> None:
    _to_top(page, "How it changes with temperature")
    _pause(page, 600)
    _hover_chart(page, 0, steps=40, frac=(0.05, 0.6))


def act_hydrogen_700(page: Page) -> None:
    _set_temperature(page, "Temperature (°C)", "700")
    _show_plain(page)


def act_co(page: Page) -> None:
    _choose_preset(page, "Reduce wüstite (FeO) to iron with carbon monoxide")
    _set_temperature(page, "Temperature (°C)", "1000")
    _show_plain(page)


def act_boudouard(page: Page) -> None:
    _choose_preset(page, "Boudouard reaction")
    _show_plain(page)
    _pause(page, 4000)
    _to_top(page, "How it changes with temperature")
    _hover_chart(page, 0, steps=30, frac=(0.05, 0.35))


def act_units(page: Page) -> None:
    side = page.locator('[data-testid="stSidebar"]')
    side.get_by_text("K", exact=True).click()
    _pause(page, 2500)
    side.get_by_text("°C", exact=True).click()


def act_substances(page: Page) -> None:
    _scroll_top(page)
    page.get_by_role("tab", name="Substances").click()
    _pause(page, 800)
    _set_temperature(page, "Heat up to (°C)", "1600")
    _scroll_to(page, "Energy to heat one tonne")


def act_advanced(page: Page) -> None:
    page.get_by_text("Show advanced properties", exact=False).first.click()
    _pause(page, 600)
    page.get_by_text("Heat capacity (Cp)", exact=True).click()
    _pause(page, 600)
    _hover_chart(page, 0, steps=50)


def act_outro(page: Page) -> None:
    _scroll_top(page)
    page.get_by_role("tab", name="Start here").click()
    _pause(page, 500)


SEGMENTS: list[tuple[str, Callable[[Page], None]]] = [
    (
        "This is pyrothermo, a thermodynamics tool for iron and steelmaking. It answers a "
        "simple question: can this reaction happen, and under what conditions? Every answer "
        "comes twice: first in plain language, then as the numbers a specialist expects.",
        act_intro,
    ),
    (
        "The start here page is a short tutorial: three steps to a first answer, a glossary "
        "of the terms used, and a clear list of what the tool cannot tell you.",
        act_tutorial,
    ),
    (
        "On the reactions tab you pick a reaction from a list. Here, reducing wustite to iron "
        "with hydrogen, the key step in hydrogen-based ironmaking. At one thousand degrees "
        "Celsius, the plain language summary says it does not go forward on its own, but it "
        "will, as long as the gas contains more than about seventy six percent hydrogen. It "
        "also says the reaction absorbs heat.",
        act_hydrogen,
    ),
    (
        "Next to it are the specialist numbers: enthalpy, entropy, Gibbs energy and log K. "
        "Below, the charts show how the reaction changes with temperature. The dot marks the "
        "chosen temperature.",
        act_specialist,
    ),
    (
        "Change the temperature and everything updates. At seven hundred degrees, the gas "
        "needs about eighty five percent hydrogen, so hydrogen reduction gets easier as it "
        "gets hotter.",
        act_hydrogen_700,
    ),
    (
        "Compare carbon monoxide, as used in a blast furnace. At one thousand degrees it needs "
        "more than about eighty four percent CO, and unlike hydrogen, it gets harder as the "
        "temperature rises.",
        act_co,
    ),
    (
        "The Boudouard reaction, carbon dioxide plus carbon giving carbon monoxide, has a "
        "turning point, and the tool finds it automatically: about seven hundred degrees "
        "Celsius. Above that, carbon regenerates the reducing gas.",
        act_boudouard,
    ),
    ("Engineers can switch between Celsius and Kelvin in the sidebar.", act_units),
    (
        "The substances tab shows how materials behave when heated. Heating one tonne of iron "
        "from room temperature to sixteen hundred degrees, including melting, takes at least "
        "about three hundred and seventy five kilowatt hours, before any furnace losses.",
        act_substances,
    ),
    (
        "Specialists can switch on the advanced properties. Here is the heat capacity of "
        "iron, with its magnetic peak and its phase changes marked.",
        act_advanced,
    ),
    (
        "Every number comes from published NIST data and is tested against the JANAF tables, "
        "and the tool states its limits: pure substances, equilibrium only, no slags and no "
        "kinetics. It is the foundation for an Ellingham diagram, an equilibrium solver and an "
        "ironmaking calculator.",
        act_outro,
    ),
]


# ---------------------------------------------------------------- pipeline


def _duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return float(json.loads(out.stdout)["format"]["duration"])


async def _synthesise(texts: list[str]) -> None:
    for i, text in enumerate(texts):
        await edge_tts.Communicate(text, VOICE, rate=RATE).save(str(BUILD / f"line{i:02d}.mp3"))


def _wait_for_server(timeout_s: float = 60.0) -> None:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2) as r:
                if r.read() == b"ok":
                    return
        except OSError:
            time.sleep(0.5)
    raise SystemExit("Streamlit did not start")


def main() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg/ffprobe not found on PATH")
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("1/4 narration")
    asyncio.run(_synthesise([t for t, _ in SEGMENTS]))
    durations = [_duration(BUILD / f"line{i:02d}.mp3") for i in range(len(SEGMENTS))]

    print("2/4 starting app")
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
         "--server.headless", "true", "--server.port", str(PORT),
         "--theme.base", "light", "--browser.gatherUsageStats", "false",
         "--client.toolbarMode", "viewer"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    starts: list[float] = []
    try:
        _wait_for_server()
        print("3/4 recording")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            ctx = browser.new_context(
                viewport=VIEW, record_video_dir=str(BUILD), record_video_size=VIEW
            )
            page = ctx.new_page()
            t0 = time.monotonic()
            page.goto(URL)
            page.get_by_role("tab", name="Start here").wait_for(timeout=60_000)
            _pause(page, 1500)
            lead = time.monotonic() - t0
            for (text, action), dur in zip(SEGMENTS, durations, strict=True):
                start = time.monotonic() - t0
                starts.append(start)
                print(f"   {start:6.1f}s  {text[:60]}...")
                action(page)
                remaining = start + dur + GAP_S - (time.monotonic() - t0)
                if remaining > 0:
                    _pause(page, int(remaining * 1000))
            _pause(page, 1500)
            video = page.video
            ctx.close()
            browser.close()
            assert video is not None
            webm = Path(video.path())
    finally:
        server.terminate()

    print("4/4 muxing")
    trim = max(0.0, lead - 0.5)  # cut the blank page-load at the start
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{trim:.2f}", "-i", str(webm)]
    filters = []
    for i in range(len(SEGMENTS)):
        cmd += ["-i", str(BUILD / f"line{i:02d}.mp3")]
        delay_ms = int(max(0.0, starts[i] - trim) * 1000)
        filters.append(f"[{i + 1}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(SEGMENTS)))
    filters.append(f"{mix}amix=inputs={len(SEGMENTS)}:normalize=0[aout]")
    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(OUT),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    print(f"done: {OUT} ({_duration(OUT):.0f} s)")


if __name__ == "__main__":
    main()
