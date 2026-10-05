require "test_helper"

module EstimatorAi
  class ClientTest < ActiveSupport::TestCase
    URL = "http://ai-test/api/v1/estimate".freeze

    setup do
      @request = EstimationRequest.new(
        transcription: "We need a customer portal with invoices and Stripe payments.",
        hourly_rate: 50
      )
      @client = Client.new(base_url: "http://ai-test")
    end

    def stub_api(status:, body:)
      stub_request(:post, URL).to_return(
        status: status, body: body.to_json, headers: { "Content-Type" => "application/json" }
      )
    end

    test "returns the parsed payload on 200 and sends the request body" do
      stub = stub_request(:post, URL)
        .with(body: { transcription: @request.transcription, hourly_rate: 50 }.to_json)
        .to_return(status: 200, body: estimation_payload.to_json,
                   headers: { "Content-Type" => "application/json" })

      payload = @client.estimate(@request)

      assert_requested stub
      assert_equal "v1", payload["prompt_version"]
      assert_equal 14_900, payload.dig("result", "summary", "total_cost_usd")
    end

    test "propagates cached true" do
      stub_api(status: 200, body: estimation_payload(cached: true))
      assert_equal true, @client.estimate(@request)["cached"]
    end

    %w[moderation prompt_injection pii].each do |reason|
      test "400 with top-level reason #{reason} raises GuardrailViolation" do
        stub_api(status: 400, body: { reason: reason, message: "Safe message for #{reason}." })

        error = assert_raises(Client::GuardrailViolation) { @client.estimate(@request) }
        assert_equal reason, error.reason
        assert_equal "Safe message for #{reason}.", error.message
      end
    end

    test "400 with an unknown reason raises InvalidRequest" do
      stub_api(status: 400, body: { reason: "weird", message: "nope" })
      assert_raises(Client::InvalidRequest) { @client.estimate(@request) }
    end

    test "422 builds a readable message from the detail list" do
      stub_api(status: 422, body: { detail: [
        { loc: [ "body", "hourly_rate" ], msg: "Input should be a valid integer", type: "int_parsing" },
        { loc: [ "body", "transcription" ], msg: "Field required", type: "missing" }
      ] })

      error = assert_raises(Client::InvalidRequest) { @client.estimate(@request) }
      assert_includes error.message, "body.hourly_rate: Input should be a valid integer"
      assert_includes error.message, "body.transcription: Field required"
    end

    test "502 raises ServerError without echoing the body" do
      stub_api(status: 502, body: { detail: "SECRET upstream failure" })

      error = assert_raises(Client::ServerError) { @client.estimate(@request) }
      assert_not_includes error.message, "SECRET"
    end

    test "500 raises ServerError with the status only" do
      stub_request(:post, URL).to_return(status: 500, body: "Traceback: SECRET")

      error = assert_raises(Client::ServerError) { @client.estimate(@request) }
      assert_equal "unexpected status 500", error.message
    end

    test "raises ArgumentError for an invalid request and makes no call" do
      invalid = EstimationRequest.new(transcription: "", hourly_rate: 50)

      assert_raises(ArgumentError) { @client.estimate(invalid) }
      assert_not_requested :post, URL
    end
  end
end
