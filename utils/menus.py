__author__ = "Se Hoon Kim(sehoon787@korea.ac.kr)"


########################
# Topic selection menu #
########################
def topic_selection_menu() -> int:
    print("\n[Options]")
    print("1. 위 결과로 계속 진행")
    print("2. 위 결과 추가 수정")
    print("3. 주제 재설정")
    return input("👉 번호를 입력하세요 (1/2/3): ").strip()


####################
# Data loader menu #
####################
def select_data_loader_menu() -> int:
    print("[Options] 데이터 처리 방식을 선택하세요:")
    print("1. 로컬 파일 업로드 (CSV 또는 SQLite)")
    print("2. 주제 기반 샘플 데이터 자동 생성")
    print("3. 관련 데이터 다운로드 링크 추천 → 다운로드 후 파일 경로 입력")
    return input("👉 번호를 입력하세요 (1/2/3): ").strip()


def select_edit_col_menu() -> int:
    print("\n✅ 위 컬럼들을 삭제하시겠습니까?")
    print("1. 예 (추천 목록 삭제)")
    print("2. 아니오 (직접 선택)")
    print("3. 건너 뛰기")
    return input("👉 선택 (1/2/3): ").strip()
