import logging

import httpx
from langchain_core.tools import tool

from .config import LARAVEL_BASE_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


def _clean_params(params: dict | None = None) -> dict:
    return {key: value for key, value in (params or {}).items() if value is not None}


def _get(path: str, params: dict | None = None) -> dict:
    """Helper untuk GET ke Laravel public API."""
    url = f"{LARAVEL_BASE_URL.rstrip('/')}{path}"
    logger.info("[CHATBOT TOOL] GET %s params=%s", path, _clean_params(params))

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = client.get(url, params=_clean_params(params))
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            return {"status": "error", "message": f"Gagal mengambil data: {str(exc)}"}


@tool
def get_sports() -> dict:
    """
    Ambil daftar semua cabang olahraga yang dipertandingkan di Tel-U Cup
    beserta kategori (putra, putri, campuran).
    Gunakan tool ini ketika user bertanya tentang cabang olahraga apa saja
    yang ada, atau ketika ingin filter pertanyaan lain berdasarkan cabor.
    """
    return _get("/sports")


@tool
def get_contingents(sport_id: int | None = None) -> dict:
    """
    Ambil daftar kontingen / tim peserta. Bisa difilter per cabang olahraga.

    Args:
        sport_id: ID cabang olahraga (opsional). Jika tidak diisi,
                  kembalikan semua kontingen.
    """
    return _get("/contingents", {"sport_id": sport_id})


@tool
def get_teams(sport_id: int | None = None, contingent_id: int | None = None) -> dict:
    """
    Ambil daftar tim terverifikasi per cabang olahraga atau kontingen.
    Response hanya berisi nama tim/kontingen, cabang olahraga, kategori,
    dan status registrasi. Tidak berisi data personal pemain.

    Args:
        sport_id: ID cabang olahraga (opsional)
        contingent_id: ID kontingen (opsional)
    """
    return _get("/teams", {"sport_id": sport_id, "contingent_id": contingent_id})


@tool
def get_match_schedule(
    sport_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> dict:
    """
    Ambil jadwal pertandingan yang akan datang.

    Args:
        sport_id: ID cabang olahraga (opsional)
        date_from: tanggal awal format YYYY-MM-DD (opsional)
        date_to: tanggal akhir format YYYY-MM-DD (opsional)
        limit: maksimal jumlah pertandingan yang dikembalikan, default 20
    """
    return _get(
        "/matches/schedule",
        {
            "sport_id": sport_id,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
        },
    )


@tool
def get_match_results(
    sport_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> dict:
    """
    Ambil hasil pertandingan yang sudah selesai lengkap dengan skor.

    Args:
        sport_id: ID cabang olahraga (opsional)
        date_from: tanggal awal format YYYY-MM-DD (opsional)
        date_to: tanggal akhir format YYYY-MM-DD (opsional)
        limit: maksimal jumlah pertandingan yang dikembalikan, default 20
    """
    return _get(
        "/matches/results",
        {
            "sport_id": sport_id,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
        },
    )


@tool
def get_bracket(sport_id: int) -> dict:
    """
    Ambil struktur bracket / bagan pertandingan untuk satu cabang olahraga.
    Berisi info tim mana yang lolos ke babak mana.

    Args:
        sport_id: ID cabang olahraga (wajib)
    """
    return _get("/brackets", {"sport_id": sport_id})


@tool
def get_venues() -> dict:
    """
    Ambil daftar Sport Center / lokasi pertandingan beserta cabor
    yang diadakan di tiap venue.
    """
    return _get("/venues")


ALL_TOOLS = [
    get_sports,
    get_contingents,
    get_teams,
    get_match_schedule,
    get_match_results,
    get_bracket,
    get_venues,
]
