__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"


########################
# Topic selection menu #
########################
def topic_selection_menu() -> str:
    return ("\n[Options]\n"
            "1. 위 결과로 계속 진행\n"
            "2. 위 결과 추가 수정\n"
            "3. 주제 재설정\n"
            "👉 번호를 입력하세요 (1/2/3): \n")


####################
# Data loader menu #
####################
def select_data_loader_menu() -> str:
    print("[Options] 데이터 처리 방식을 선택하세요:")
    print("1. 로컬 파일 업로드 (CSV 또는 SQLite)")
    print("2. 주제 기반 샘플 데이터 자동 생성")
    return input("👉 번호를 입력하세요 (1/2): ").strip()


def select_edit_col_menu() -> str:
    return ("\n✅ 위 컬럼들을 삭제하시겠습니까?\n"
            "1. 예 (추천 목록 삭제)\n"
            "2. 아니오 (직접 선택)"
            "3. 건너 뛰기\n"
            "👉 선택 (1/2/3): \n")
