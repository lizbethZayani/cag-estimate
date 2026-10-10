require "test_helper"

class MessagesControllerTest < ActionDispatch::IntegrationTest
  setup do
    @chat_session = ChatSession.create!(remote_session_id: "sess-1")
    @url = "#{Rails.application.config.estimator_ai.base_url}/api/v1/sessions/sess-1/estimate"
    @params = {
      session_estimation_request: {
        transcript: "We need a customer portal with invoices and Stripe payments.",
        hourly_rate: 50, tier: ""
      }
    }
  end

  def stub_estimate(status: 200, body: session_estimation_payload)
    stub_request(:post, @url).to_return(
      status: status, body: body.to_json, headers: { "Content-Type" => "application/json" }
    )
  end

  def post_message(params = @params)
    post chat_session_messages_path(@chat_session), params: params
  end

  test "a valid message persists an estimation linked to the session and refreshes the mirror" do
    stub_estimate(body: session_estimation_payload.merge("turns" => 3))

    assert_difference "Estimation.count", 1 do
      post_message
    end

    assert_redirected_to chat_session_path(@chat_session)
    estimation = Estimation.order(:id).last
    assert_equal @chat_session, estimation.chat_session
    assert_equal @params[:session_estimation_request][:transcript], estimation.transcription
    assert_equal 50, estimation.hourly_rate
    assert_equal "v2", estimation.prompt_version
    assert_equal false, estimation.cached
    assert_equal "developer", estimation.response_payload["resolved_tier"]

    @chat_session.reload
    assert_equal 3, @chat_session.turn_count
    assert_equal "Customer Portal", @chat_session.latest_metadata["project_name"]
    assert_equal [ "Rails" ], @chat_session.latest_metadata["mentioned_technologies"]
  end

  test "the tier is sent only when chosen" do
    stub_estimate
    post_message
    assert_requested(:post, @url) { |req| !req.body.include?("tier=") }

    WebMock.reset!
    stub_estimate
    post_message(session_estimation_request: @params[:session_estimation_request].merge(tier: "executive"))
    assert_requested(:post, @url) { |req| req.body.include?("tier=executive") }
  end

  test "an invalid tier is rejected locally" do
    assert_no_difference "Estimation.count" do
      post_message(session_estimation_request: @params[:session_estimation_request].merge(tier: "ceo"))
    end
    assert_response :unprocessable_entity
    assert_not_requested :post, @url
  end

  test "an attachment reaches the API as a multipart part" do
    stub_estimate(body: session_estimation_payload(attachments: [ "brief.txt" ]))
    file = Rack::Test::UploadedFile.new(StringIO.new("Scope: invoices"), "text/plain", original_filename: "brief.txt")

    post_message(@params.merge(attachments: [ file ]))

    assert_redirected_to chat_session_path(@chat_session)
    assert_requested(:post, @url) do |req|
      req.headers["Content-Type"].start_with?("multipart/form-data") &&
        req.body.include?('name="attachments"; filename="brief.txt"') &&
        req.body.include?("Scope: invoices")
    end
    assert_equal [ "brief.txt" ], Estimation.order(:id).last.response_payload["attachments"]
  end

  test "invalid params return 422, keep the typed values and persist nothing" do
    assert_no_difference "Estimation.count" do
      post_message(session_estimation_request: { transcript: "", hourly_rate: 5 })
    end
    assert_response :unprocessable_entity
    assert_not_requested :post, @url
    assert_select "form[action='#{chat_session_messages_path(@chat_session)}']"
    assert_select "input[name='session_estimation_request[hourly_rate]'][value='5']"
    assert_select "[data-panel='memory']"
  end

  test "a guardrail 400 re-renders with the API message and does not persist" do
    stub_estimate(status: 400, body: { reason: "pii", message: "Please remove personal data." })

    assert_no_difference "Estimation.count" do
      post_message
    end
    assert_response :unprocessable_entity
    assert_select "[role=alert][data-kind='rejected']", text: /Please remove personal data/
    assert_select "textarea", text: /customer portal with invoices/
    assert_equal 0, @chat_session.reload.turn_count
  end

  test "an API 422 re-renders with a flash and does not persist" do
    stub_estimate(status: 422, body: { detail: "bad" })
    assert_no_difference "Estimation.count" do
      post_message
    end
    assert_response :unprocessable_entity
    assert_select "[role=alert][data-kind='rejected']"
  end

  test "a lost remote session deletes the mirror and starts over" do
    stub_request(:get, %r{/api/v1/sessions/sess-1\z}).to_return(
      status: 200, body: { max_turns: 6 }.to_json, headers: { "Content-Type" => "application/json" }
    )
    get chat_session_path(@chat_session) # remembers the conversation in the Rails session
    stub_estimate(status: 404, body: { detail: "Session not found" })

    assert_difference "ChatSession.count", -1 do
      post_message
    end
    assert_redirected_to new_chat_session_path
    assert_equal "The conversation expired. A new one was started.", flash[:alert]
    assert_equal 0, Estimation.count

    # the Rails session key was cleared: a fresh remote session is created
    create = stub_request(:post, "#{Rails.application.config.estimator_ai.base_url}/api/v1/sessions")
             .to_return(status: 201, body: { session_id: "sess-2" }.to_json,
                        headers: { "Content-Type" => "application/json" })
    get new_chat_session_path
    assert_requested create, times: 1
  end

  test "an API 502 returns 503 with a generic message and does not persist" do
    stub_estimate(status: 502, body: { detail: "SECRET upstream" })

    assert_no_difference "Estimation.count" do
      post_message
    end
    assert_response :service_unavailable
    assert_select "[role=alert][data-kind='unavailable']", text: /AI service unavailable/
    assert_no_match "SECRET", response.body
  end

  test "a timeout returns 503 without leaking the raw error" do
    stub_request(:post, @url).to_timeout
    assert_no_difference "Estimation.count" do
      post_message
    end
    assert_response :service_unavailable
    assert_match "AI service unavailable", response.body
    assert_no_match(/execution expired|timeout/i, response.body.sub("AI service unavailable", ""))
  end

  test "a connection failure returns 503" do
    stub_request(:post, @url).to_raise(Faraday::ConnectionFailed.new("refused SECRET"))
    post_message
    assert_response :service_unavailable
    assert_no_match "SECRET", response.body
  end

  test "an unknown conversation redirects to new" do
    post chat_session_messages_path(chat_session_id: 987_654), params: @params
    assert_redirected_to new_chat_session_path
  end
end
