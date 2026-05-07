id
  train.csv의 각 row를 구별하기 위한 ID. 학습에 입력데이터로 사용하지 않음.

site_token
  임의의 사이트를 나타내는 문자열. site_12345678 형식이다. test.csv에도 그대로 등장하므로 입력데이터로 쓰임.
  앞의 'site_'를 빼고 뒤의 고유문자열을 훈련에 쓰면 좋을듯? test.csv에 96개의 고유한 site_token이 있다고 함.

  train.csv파일 내에서는 같은 site_token이라도 서로 다른 cleaned_html와 candidate_elements를 가짐. 그냥 같은 사이트/도메인/업무군 정도의 힌트만.
  *site_token별로 임베딩하면 site_token마다 다른 성격을 가지고 있으니 이것도 feature가 될 수 있지 않을까?

  *특이 토큰: site_2aa627db 얘는 train.csv파일에만 있는 토큰임. 특히 거의 CLICK만 있어서 CLICk 관련된 학습을 조금 왜곡할 가능성이 있음.
  제외했을 때와 하지 않았을 때를 비교하자.

task
  웹 에이전트의 최종 목적.
  생각보다 중복된 Task가 많다.

  즉 같은 목표라도 진행 단계가 다르면 정답 target이 달라집니다.
  예를 들면 “호텔을 예약하라”라는 같은 task가 29번 등장하고, history step이 0 -> 1 -> 2 -> ... -> 28로 증가하면서 다음에
  눌러야 할 target이 계속 바뀝니다.

  train과 test 사이에 site_token과 task가 둘 다 동일한 row는 없다. site_token이 같다면 task는 다르다는뜻.


history
  Task를 이루기 위해 지금까지 수행한 것들의 Sequence.
  [태그명] 텍스트 -> op 형식임. 다만 history에 나오는 op는 CLICK/TYPE/SELECT 이외에 HOVER/ENTER 도 있음.
  CLICK   24,341
  TYPE    10,909
  SELECT   5,283
  HOVER      454
  ENTER       19
  candidate_elements의 tag와 이름이 조금 다릅니다. 예를 들어 history에는 textbox, link, combobox가 나오지만 candidate에는
  주로 input, a, select로 나옵니다. 이 둘을 매핑하는 feature가 있으면 도움이 됩니다.
  history는 어디까지 완료했는지 판단하는 용도에 가깝다. 따라서 target_id 예측에 매우 중요함.

cleaned_html
  두 가지 타입이 있음. 1.구조화된 html 2.실제 웹 계열 html
  1. 구조화된 html
    <h1>Cafeteria menu update</h1>
      <aside class="workflow-context">
        current step 6 of 7
      </aside>
      <aside class="completed-fields">
        Completed: New menu update, Menu item, Station
      </aside>
      <section aria-label="current workflow panel">
        ...
      </section>
    꽤 쉬움. current_step = history_step+1이고, completed_fields = history_step임.(100%)
    이 형식에서는 정답에 가까운 진행상태 힌트가 들어있음.
    completed-fields에 들어간 candidate는 target이 아님.(100%)

  2. 실제 웹 계열 html
    <html>
      ...
    </html>
    여러가지 노이즈?가 섞인 형태라고 볼 수 있음. candidate의 text, attrs가 비어 있는 경우가 많음.
    여기 안에서 target candidate를 찾는 것이 중요 과제일듯
    결론적으로, raw web HTML은 workflow 구조 힌트가 없지만 history가 긴 row가 많습니다.
    따라서 raw web HTML 쪽 target_id 개선은 cleaned_html 자체보다도 history sequence를 잘 쓰는 쪽이 더 중요할 가능성이 큽니
    다.
    candidate_elements과의 관계: candidate_elemenets는 raw HTML에서 추출된 후보 요약본이다.
    하지만 candidate_id/backend_node_id 연결고리가 제거되어 있다.
    따라서 HTML에서 후보를 1:1로 정확히 복원하기 어렵다.
    text/attrs/tag 매칭은 보조 신호로만 쓸 수 있다.

  site_token은 동일한 row가 자주 등장하지만 workflow/raw Web이 섞여서 나오지는 않는다.
  즉, site_token 만으로 html 구조를 판별할 수 있다. 따라서 workflow web과 raw web을 분리해서
  target_id 예측 전략을 다르게 가져가야 한다.

candidate_elements
  15개의 후보를 json 형식으로 표현해 놓음. 그 중에 하나가 정답임. 다만 raw web html의 경우,
  text나 attrs 속성이 비어있는 경우도 있다.

op
  CLICK/SELECT/TYPE 셋 중 하나임. op가 CLICK일 경우 Value는 NULL이고, SELECT나 TYPE일 경우 Value 는 NOTNULL

target_id
  candiate_elements 중 반드시 정답이 하나가 있다.

value
  op가 TYPE이나 SELECT일 경우 실제로 입력해야 하는 값. 대부분 Task안에 정보가 있다.
  다만 표기 차이 때문에 정확한 matching은 안된다. normalize가 필요할듯.
  대소문자를 비교하는지, 띄어쓰기를 엄격하게 보는지 체크가 필요함.

  예시:
  task: return at 6pm
  value: 6 00 PM
  task: price high to low
  value: Price High - Low
  task: 123 st rd
  value: 123st rd

  분석해 보니 일부 문제에서 값이 조금 이상하게 들어간 경우가 있었음.
예를 들면:
B-393 about door latch stuck
정답은 방 번호만 필요한데, 뒤에 설명까지 같이 들어갔음
정답에 가까운 값은:
B-393
또 다른 예시는:
'queue drain stalled'
이 경우 따옴표까지 들어갔지만, 실제로는 안의 내용만 필요함.
queue drain stalled
그래서 기존 제출 파일을 기본 답안으로 두고,
확실히 고칠 수 있는 부분만 수정하는 somenna_pipeline 파일 제작

찾아본결과 test.csv의 문제는 두 종류로 나눠져 있음.
1. 깔끔한 문제
예를 들면:
이름 칸
날짜 칸
우선순위 선택 칸
제출 버튼

2. 복잡한 문제
이 문제들은 버튼과 링크가 많고,
후보들의 이름이 비어 있는 경우도 많음.

후보 1: 이름 없음
후보 2: 이름 없음
후보 3: role=button

이런 문제는 규칙만으로 맞히기 어려워보임
나중에 로컬 LLM을 붙여서 학습시켜서 판단하게 해보는 것도?

somenna_pipeline은 쉬운 문제 쪽을 먼저 품.
즉, data-site=가 있는 깔끔한 문제를 찾아서
train.csv에서 흔히 보이는 규칙으로 다시 품.
그리고 어려운 실제 웹사이트 문제는 건드리지 않음.
대신 기존에 점수가 좋았던 somenna_submission (4).csv의 답을 그대로 사용(점수 보존)


얼마나 잘 맞추는지 train.csv로 검증한 결과
가상으로 생성한 synthetic 문제에서
1067개 중 1067개를 맞춤. (정답률 100%)

*단, 가상으로 생성한 문제라 신뢰가 있을지는 추후 검증 필요