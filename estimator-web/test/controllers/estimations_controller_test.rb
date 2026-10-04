require "test_helper"

class EstimationsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @url = "#{Rails.application.config.estimator_ai.base_url}/api/v1/estimate"
    @params = {
      estimation_request: {
        transcription: "We need a customer portal with invoices and Stripe payments.",
        hourly_rate: 50
      }
    }
  end

  def stub_api(status:, body:)
    stub_request(:post, @url).to_return(
      status: status, body: body.to_json, headers: { "Content-Type" => "application/json" }
    )
  end

  test "GET new renders the form" do
    get new_estimation_path
    assert_response :success
    assert_select "textarea[name='estimation_request[transcription]']"
    assert_select "input[name='estimation_request[hourly_rate]']"
  end

  test "root is the new form" do
    get root_path
    assert_response :success
  end

  test "POST create persists and redirects to show" do
    stub_api(status: 200, body: estimation_payload)

    assert_difference "Estimation.count", 1 do
      post estimations_path, params: @params
    end

    estimation = Estimation.order(:id).last
    assert_redirected_to estimation_path(estimation)
    assert_equal 50, estimation.hourly_rate
    assert_equal "v1", estimation.prompt_version
    assert_equal false, estimation.cached
  end

  test "POST create persists the cached flag" do
    stub_api(status: 200, body: estimation_payload(cached: true))
    post estimations_path, params: @params
    assert Estimation.order(:id).last.cached
  end

  test "GET show renders totals and tasks" do
    stub_api(status: 200, body: estimation_payload)
    post estimations_path, params: @params
    follow_redirect!

    assert_response :success
    assert_match "Customer Portal &amp; Invoice Management System", response.body
    assert_match "298", response.body
    assert_match "14,900", response.body
    assert_match "User Authentication &amp; Session Management", response.body
  end

  test "GET index lists saved estimations" do
    Estimation.create!(transcription: "Portal for invoices and payments", hourly_rate: 50,
                       response_payload: estimation_payload, prompt_version: "v1")
    get estimations_path
    assert_response :success
    assert_match "Portal for invoices", response.body
  end

  test "invalid params return 422 and persist nothing" do
    assert_no_difference "Estimation.count" do
      post estimations_path, params: { estimation_request: { transcription: "", hourly_rate: 5 } }
    end
    assert_response :unprocessable_entity
    assert_not_requested :post, @url
  end

  test "guardrail 400 re-renders the form with the API message" do
    stub_api(status: 400, body: { reason: "pii", message: "Please remove personal data." })

    assert_no_difference "Estimation.count" do
      post estimations_path, params: @params
    end
    assert_response :unprocessable_entity
    assert_match "Please remove personal data.", response.body
  end

  test "API 422 re-renders the form" do
    stub_api(status: 422, body: { detail: [ { loc: [ "body", "x" ], msg: "bad", type: "t" } ] })
    post estimations_path, params: @params
    assert_response :unprocessable_entity
    assert_match "body.x: bad", response.body
  end

  test "API 502 returns 503 with a generic message" do
    stub_api(status: 502, body: { detail: "SECRET upstream" })

    assert_no_difference "Estimation.count" do
      post estimations_path, params: @params
    end
    assert_response :service_unavailable
    assert_match "AI service unavailable", response.body
    assert_no_match "SECRET", response.body
  end

  test "timeout returns 503 without leaking the raw error" do
    stub_request(:post, @url).to_timeout
    post estimations_path, params: @params
    assert_response :service_unavailable
    assert_match "AI service unavailable", response.body
    assert_no_match(/execution expired|timeout/i, response.body.sub("AI service unavailable", ""))
  end

  test "connection failure returns 503" do
    stub_request(:post, @url).to_raise(Faraday::ConnectionFailed.new("refused SECRET"))
    post estimations_path, params: @params
    assert_response :service_unavailable
    assert_no_match "SECRET", response.body
  end
end
