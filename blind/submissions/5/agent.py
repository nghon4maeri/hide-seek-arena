from __future__ import annotations

import sys
from collections import Counter, deque
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

DUONG_DAN_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(DUONG_DAN_SRC) not in sys.path:
    sys.path.insert(0, str(DUONG_DAN_SRC))

from agent_interface import GhostAgent as LopGhostCoSo
from agent_interface import PacmanAgent as LopPacmanCoSo
from environment import Move as HuongDi

LEN = HuongDi.UP
XUONG = HuongDi.DOWN
TRAI = HuongDi.LEFT
PHAI = HuongDi.RIGHT
DUNG_YEN = HuongDi.STAY

CAC_HUONG_CHINH = (LEN, XUONG, TRAI, PHAI)
HUONG_THEO_DO_LECH = {
    (-1, 0): LEN,
    (1, 0): XUONG,
    (0, -1): TRAI,
    (0, 1): PHAI,
}

def _nam_trong_ban_do(vi_tri: tuple[int, int], ban_do: np.ndarray) -> bool:
    """Kiểm tra một vị trí có nằm trong giới hạn bản đồ hay không."""
    hang, cot = vi_tri
    return 0 <= hang < ban_do.shape[0] and 0 <= cot < ban_do.shape[1]

def _vi_tri_sau_khi_di(
    vi_tri: tuple[int, int], huong: HuongDi
) -> tuple[int, int]:
    """Tính vị trí mới sau khi đi một ô theo hướng đã chọn."""
    thay_doi_hang, thay_doi_cot = huong.value
    return vi_tri[0] + thay_doi_hang, vi_tri[1] + thay_doi_cot

def _la_o_trong_da_biet(vi_tri: tuple[int, int], ban_do: np.ndarray) -> bool:
    """Chỉ cho phép đi vào ô trống đã quan sát, tức ô có giá trị 0."""
    return _nam_trong_ban_do(vi_tri, ban_do) and ban_do[vi_tri] == 0

def _cac_o_ke_da_biet(
    vi_tri: tuple[int, int], ban_do: np.ndarray
) -> list[tuple[HuongDi, tuple[int, int]]]:
    """Lấy các ô kề có thể đi an toàn từ vị trí hiện tại."""
    ket_qua: list[tuple[HuongDi, tuple[int, int]]] = []

    for huong in CAC_HUONG_CHINH:
        vi_tri_moi = _vi_tri_sau_khi_di(vi_tri, huong)
        if _la_o_trong_da_biet(vi_tri_moi, ban_do):
            ket_qua.append((huong, vi_tri_moi))

    return ket_qua

def _cap_nhat_ban_do_nho(
    ban_do_nho: Optional[np.ndarray], quan_sat_moi: np.ndarray
) -> np.ndarray:
    """Ghép vùng vừa nhìn thấy vào bản đồ đã ghi nhớ từ các bước trước."""
    if ban_do_nho is None or ban_do_nho.shape != quan_sat_moi.shape:
        ban_do_nho = np.full(quan_sat_moi.shape, -1, dtype=np.int8)

    o_da_nhin_thay = quan_sat_moi != -1
    ban_do_nho[o_da_nhin_thay] = quan_sat_moi[o_da_nhin_thay]
    return ban_do_nho

def _tim_duong_bfs(
    ban_do: np.ndarray,
    vi_tri_dau: tuple[int, int],
    cac_dich: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Tìm đường ngắn nhất bằng BFS trên các ô đã biết là an toàn.

    Kết quả không chứa vị trí bắt đầu. Nếu không tìm được đường thì trả về [].
    """
    tap_dich = set(cac_dich)
    if not tap_dich or vi_tri_dau in tap_dich:
        return []

    hang_doi = deque([vi_tri_dau])
    cha: dict[tuple[int, int], Optional[tuple[int, int]]] = {vi_tri_dau: None}
    dich_da_toi: Optional[tuple[int, int]] = None

    while hang_doi:
        vi_tri_hien_tai = hang_doi.popleft()

        for _, vi_tri_ke in _cac_o_ke_da_biet(vi_tri_hien_tai, ban_do):
            if vi_tri_ke in cha:
                continue

            cha[vi_tri_ke] = vi_tri_hien_tai

            if vi_tri_ke in tap_dich:
                dich_da_toi = vi_tri_ke
                hang_doi.clear()
                break

            hang_doi.append(vi_tri_ke)

    if dich_da_toi is None:
        return []

    duong_di_nguoc: list[tuple[int, int]] = []
    con_tro: Optional[tuple[int, int]] = dich_da_toi

    while con_tro is not None and con_tro != vi_tri_dau:
        duong_di_nguoc.append(con_tro)
        con_tro = cha[con_tro]

    duong_di_nguoc.reverse()
    return duong_di_nguoc


def _huong_di_giua_hai_o(
    o_dau: tuple[int, int], o_sau: tuple[int, int]
) -> HuongDi:
    """Đổi chênh lệch tọa độ giữa hai ô kề thành hướng di chuyển."""
    do_lech = o_sau[0] - o_dau[0], o_sau[1] - o_dau[1]
    return HUONG_THEO_DO_LECH[do_lech]

def _cac_o_bien_kham_pha(ban_do: np.ndarray) -> set[tuple[int, int]]:
    """Tìm các ô trống đã biết nhưng nằm cạnh ít nhất một ô chưa biết."""
    cac_o_bien: set[tuple[int, int]] = set()
    so_hang, so_cot = ban_do.shape

    for hang in range(so_hang):
        for cot in range(so_cot):
            if ban_do[hang, cot] != 0:
                continue

            for huong in CAC_HUONG_CHINH:
                hang_moi, cot_moi = _vi_tri_sau_khi_di((hang, cot), huong)

                if (
                    0 <= hang_moi < so_hang
                    and 0 <= cot_moi < so_cot
                    and ban_do[hang_moi, cot_moi] == -1
                ):
                    cac_o_bien.add((hang, cot))
                    break

    return cac_o_bien

def _cung_hang_cot_khong_vuong_tuong(
    ban_do: np.ndarray,
    vi_tri_mot: tuple[int, int],
    vi_tri_hai: tuple[int, int],
) -> bool:
    """Kiểm tra hai vị trí có cùng hàng/cột và không bị tường chắn hay không."""
    if vi_tri_mot[0] == vi_tri_hai[0]:
        hang = vi_tri_mot[0]
        cot_trai, cot_phai = sorted((vi_tri_mot[1], vi_tri_hai[1]))
        return not np.any(ban_do[hang, cot_trai + 1:cot_phai] == 1)

    if vi_tri_mot[1] == vi_tri_hai[1]:
        cot = vi_tri_mot[1]
        hang_tren, hang_duoi = sorted((vi_tri_mot[0], vi_tri_hai[0]))
        return not np.any(ban_do[hang_tren + 1:hang_duoi, cot] == 1)

    return False


# ============================================================
# Pacman
# ============================================================

class PacmanAgent(LopPacmanCoSo):
    """
    Pacman ghi nhớ bản đồ, đuổi theo vị trí cuối của Ghost và khám phá
    các vùng chưa biết khi mất dấu đối thủ.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.toc_do_pacman = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Pacman BFS ghi nhớ bản đồ - Nhóm 5"
        self._khoi_tao_lai_bo_nho()

    def _khoi_tao_lai_bo_nho(self) -> None:
        """Xóa dữ liệu của trận cũ trước khi bắt đầu một trận mới."""
        self.ban_do_da_biet: Optional[np.ndarray] = None
        self.vi_tri_doi_thu_cuoi: Optional[tuple[int, int]] = None
        self.vi_tri_truoc: Optional[tuple[int, int]] = None
        self.so_lan_di_qua: Counter[tuple[int, int]] = Counter()

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple[int, int],
        enemy_position: Optional[tuple[int, int]],
        step_number: int,
    ):
        if step_number == 1:
            self._khoi_tao_lai_bo_nho()

        self.ban_do_da_biet = _cap_nhat_ban_do_nho(
            self.ban_do_da_biet, map_state
        )
        self.so_lan_di_qua[my_position] += 1

        if enemy_position is not None:
            self.vi_tri_doi_thu_cuoi = enemy_position

        duong_di: list[tuple[int, int]] = []

        # Ưu tiên 1: đuổi vị trí đang thấy hoặc vị trí cuối cùng đã thấy Ghost.
        muc_tieu = enemy_position or self.vi_tri_doi_thu_cuoi
        if (
            muc_tieu is not None
            and _la_o_trong_da_biet(muc_tieu, self.ban_do_da_biet)
        ):
            duong_di = _tim_duong_bfs(
                self.ban_do_da_biet, my_position, {muc_tieu}
            )

            # Đã đến vị trí cũ mà không thấy Ghost thì bỏ mục tiêu cũ.
            if (
                not duong_di
                and enemy_position is None
                and my_position == muc_tieu
            ):
                self.vi_tri_doi_thu_cuoi = None

        # Ưu tiên 2: đi tới vùng biên để mở rộng tầm nhìn.
        if not duong_di:
            cac_o_bien = _cac_o_bien_kham_pha(self.ban_do_da_biet)
            cac_o_bien.discard(my_position)
            duong_di = _tim_duong_bfs(
                self.ban_do_da_biet, my_position, cac_o_bien
            )

        if duong_di:
            hanh_dong = self._doi_duong_di_thanh_hanh_dong(
                my_position, duong_di
            )
            self.vi_tri_truoc = my_position
            return hanh_dong

        # Không có mục tiêu rõ ràng: chọn ô an toàn và ít đi qua nhất.
        cac_lua_chon = _cac_o_ke_da_biet(my_position, self.ban_do_da_biet)
        if cac_lua_chon:
            cac_lua_chon.sort(
                key=lambda lua_chon: (
                    self.so_lan_di_qua[lua_chon[1]],
                    lua_chon[1] == self.vi_tri_truoc,
                    CAC_HUONG_CHINH.index(lua_chon[0]),
                )
            )
            huong, _ = cac_lua_chon[0]
            self.vi_tri_truoc = my_position
            return huong, 1

        return DUNG_YEN, 1

    def _doi_duong_di_thanh_hanh_dong(
        self,
        vi_tri_dau: tuple[int, int],
        duong_di: list[tuple[int, int]],
    ) -> tuple[HuongDi, int]:
        """Tận dụng tốc độ Pacman nếu đoạn đầu của đường đi là đường thẳng."""
        huong_dau = _huong_di_giua_hai_o(vi_tri_dau, duong_di[0])
        so_buoc = 1
        vi_tri_hien_tai = duong_di[0]

        for vi_tri_ke in duong_di[1:self.toc_do_pacman]:
            if _huong_di_giua_hai_o(vi_tri_hien_tai, vi_tri_ke) != huong_dau:
                break

            so_buoc += 1
            vi_tri_hien_tai = vi_tri_ke

        return huong_dau, min(so_buoc, self.toc_do_pacman)


# ============================================================
# Ghost
# ============================================================

class GhostAgent(LopGhostCoSo):
    """
    Ghost chấm điểm từng nước đi dựa trên khoảng cách, số lối thoát,
    nguy cơ ngõ cụt và khả năng bị Pacman nhìn thấy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost chấm điểm an toàn - Nhóm 5"
        self._khoi_tao_lai_bo_nho()

    def _khoi_tao_lai_bo_nho(self) -> None:
        """Xóa dữ liệu của trận cũ trước khi bắt đầu một trận mới."""
        self.ban_do_da_biet: Optional[np.ndarray] = None
        self.vi_tri_doi_thu_cuoi: Optional[tuple[int, int]] = None
        self.buoc_nhin_thay_cuoi = -10_000
        self.vi_tri_truoc: Optional[tuple[int, int]] = None
        self.so_lan_di_qua: Counter[tuple[int, int]] = Counter()

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple[int, int],
        enemy_position: Optional[tuple[int, int]],
        step_number: int,
    ) -> HuongDi:
        if step_number == 1:
            self._khoi_tao_lai_bo_nho()

        self.ban_do_da_biet = _cap_nhat_ban_do_nho(
            self.ban_do_da_biet, map_state
        )
        self.so_lan_di_qua[my_position] += 1

        if enemy_position is not None:
            self.vi_tri_doi_thu_cuoi = enemy_position
            self.buoc_nhin_thay_cuoi = step_number

        vi_tri_nguy_hiem = self.vi_tri_doi_thu_cuoi
        tuoi_thong_tin = step_number - self.buoc_nhin_thay_cuoi

        # Vị trí Pacman đã quá cũ thì không dùng để đe dọa nữa.
        if tuoi_thong_tin > 20:
            vi_tri_nguy_hiem = None

        cac_lua_chon: list[tuple[HuongDi, tuple[int, int]]] = [
            (DUNG_YEN, my_position)
        ]
        cac_lua_chon.extend(
            _cac_o_ke_da_biet(my_position, self.ban_do_da_biet)
        )

        bang_diem: list[tuple[float, int, HuongDi, tuple[int, int]]] = []

        for thu_tu, (huong, vi_tri_moi) in enumerate(cac_lua_chon):
            so_loi_thoat = len(
                _cac_o_ke_da_biet(vi_tri_moi, self.ban_do_da_biet)
            )
            diem = 3.0 * so_loi_thoat - 1.4 * self.so_lan_di_qua[vi_tri_moi]

            # Tránh quay đầu ngay và hạn chế đi vào ngõ cụt.
            if vi_tri_moi == self.vi_tri_truoc:
                diem -= 3.0
            if so_loi_thoat <= 1 and huong != DUNG_YEN:
                diem -= 7.0
            if huong == DUNG_YEN:
                diem -= 2.5

            if vi_tri_nguy_hiem is not None:
                khoang_cach = (
                    abs(vi_tri_moi[0] - vi_tri_nguy_hiem[0])
                    + abs(vi_tri_moi[1] - vi_tri_nguy_hiem[1])
                )
                do_tin_cay = max(0.25, 1.0 - tuoi_thong_tin / 25.0)

                # Càng xa Pacman thì điểm càng cao.
                diem += 8.0 * do_tin_cay * khoang_cach

                # Không nên đứng cùng hàng hoặc cột nếu không có tường che.
                if _cung_hang_cot_khong_vuong_tuong(
                    self.ban_do_da_biet, vi_tri_moi, vi_tri_nguy_hiem
                ):
                    diem -= 8.0 * do_tin_cay

                # Khoảng cách nhỏ hơn 2 là vùng có thể bị bắt ngay.
                if khoang_cach < 2:
                    diem -= 1000.0

            # Đổi thứ tự ưu tiên khi bằng điểm để tránh lặp đường máy móc.
            uu_tien_phu = (thu_tu - step_number) % len(cac_lua_chon)
            bang_diem.append((diem, -uu_tien_phu, huong, vi_tri_moi))

        _, _, huong_tot_nhat, _ = max(
            bang_diem, key=lambda muc: (muc[0], muc[1])
        )

        self.vi_tri_truoc = my_position
        return huong_tot_nhat