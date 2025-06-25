# VibeCraft

1. 사용자가 주제를 입력하면, 해당 주제는 topic_server를 통해 처리되며, Claude Code MCP와 연동되어 주제에 대한 요약 및 목적 정의가 수행.  
이를 통해 이후 데이터 수집 및 분석에 필요한 기반 정보를 구성
2. 주제가 설정된 후, 사용자가 관련 데이터를 직접 업로드하는 경우에는 data_upload_server에서 해당 데이터를 수신하고 CSV 또는 SQLite 형태로 변환하여 저장.  
만약 사용자가 데이터를 제공하지 않으면, web_search_client가 주제에 적합한 데이터를 자동으로 웹에서 수집하고, 이를 동일한 형식으로 정제하여 저장합니다.
3. 수집 또는 업로드된 데이터가 확보되면, coder_generator_client가 이를 활용하여 적절한 시각화와 콘텐츠 구성을 갖춘 웹 페이지 코드를 자동으로 생성.  
이 과정 역시 Claude 기반 코드 생성 MCP와 통합되어 수행.
4. 생성된 웹 페이지 코드는 deploy_client를 통해 Vercel MCP에 전달되어 자동으로 배포.  
배포가 완료되면 최종 URL이 생성되어 사용자가 웹 페이지를 직접 확인할 수 있도록 제공.

```python
# 1. 주제 설정
topic = topic_server.generate_topic(prompt)
# → Claude Code MCP로 주제 요약 및 목적 정의

# 2. 데이터 수집 또는 업로드
if user.uploads_data():
    data = data_upload_server.process(user.uploaded_file)
else:
    data = web_search_client.collect_and_format(topic)

# 3. 웹 코드 생성
web_code = coder_generator_client.generate(data)

# 4. 웹 페이지 배포
deploy_url = deploy_client.deploy(web_code)

# 결과 제공
return deploy_url
```

## 개요

VibeCraft 사용자가 제시한 주제를 기반으로 데이터를 수집/업로드하고, 해당 데이터를 분석하여 자동으로 웹 페이지를 생성하고 배포하는 End-to-End 시스템입니다.

## 🔁 전체 프로세스 흐름

1. **주제 설정 (Topic Server)**
   - 사용자가 입력한 prompt를 기반으로 Claude MCP를 통해 주제를 설정합니다.
   - `topic_server`는 설정된 주제를 관리하고 다음 단계로 전달합니다.

2. **데이터 수집/업로드 (Data Upload Server + Web Search Client)**
   - 사용자가 직접 데이터를 업로드하면 `data_upload_server`가 이를 CSV 또는 SQLite 형식으로 저장합니다.
   - 업로드된 데이터가 없을 경우, `web_search_client`가 주제 기반 데이터를 웹에서 수집하고 자동으로 정제 및 저장합니다.

3. **웹 페이지 코드 생성 (Coder Generator Client)**
   - 수집된 데이터를 기반으로 `coder_generator_client`는 적절한 시각화, 구조, UI를 포함한 웹 페이지 코드를 자동 생성합니다.
   - Claude Code MCP 또는 사내 코드 LLM과 통합되어 코드 품질을 보장합니다.

4. **웹 페이지 배포 (Deploy Client)**
   - 생성된 웹 페이지는 `deploy_client`를 통해 Vercel 플랫폼에 자동 배포됩니다.
   - 배포 완료 후 사용자는 배포된 웹 주소(URL)를 통해 결과를 확인할 수 있습니다.

---

## 📁 디렉토리 구조

```plaintext
project-root/
│
├── servers/
│   ├── topic_server/            # 주제 설정용 서버
│   └── data_upload_server/      # 사용자 데이터 업로드 처리
│
├── clients/
│   ├── web_search_client/       # 웹 수집 및 DB 변환 처리
│   ├── coder_generator_client/  # 웹 코드 생성 (LLM 활용)
│   └── deploy_client/           # Vercel 기반 웹 배포
│
├── utils/                      # 공통 유틸, 설정, 모델 정의
└── README.md
```
