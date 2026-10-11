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

    # --- chat sessions -----------------------------------------------------

    BASE = "http://ai-test/api/v1/sessions".freeze
    JSON_HEADERS = { "Content-Type" => "application/json" }.freeze

    def stub_json(method, url, status:, body: nil)
      stub_request(method, url).to_return(
        status: status, body: body.nil? ? "" : body.to_json, headers: JSON_HEADERS
      )
    end

    def session_request(**overrides)
      SessionEstimationRequest.new({ transcript: "Add a mobile app to the portal.", hourly_rate: 55 }.merge(overrides))
    end

    def upload(name, content, type)
      file = Tempfile.new([ "up", File.extname(name) ])
      file.write(content)
      file.rewind
      (@tempfiles ||= []) << file
      Rack::Test::UploadedFile.new(file.path, type, original_filename: name)
    end

    teardown { Array(@tempfiles).each(&:close!) }

    test "create_session returns the id on 201" do
      stub_json(:post, BASE, status: 201, body: { session_id: "sess-1" })
      assert_equal "sess-1", @client.create_session
    end

    test "create_session raises ServerError on a non-201" do
      stub_json(:post, BASE, status: 500, body: { detail: "SECRET" })
      error = assert_raises(Client::ServerError) { @client.create_session }
      assert_not_includes error.message, "SECRET"
    end

    test "get_session returns the snapshot on 200" do
      stub_json(:get, "#{BASE}/sess-1", status: 200, body: { session_id: "sess-1", turns: 2 })
      assert_equal 2, @client.get_session("sess-1")["turns"]
    end

    test "get_session raises SessionNotFound on 404" do
      stub_json(:get, "#{BASE}/gone", status: 404, body: { detail: "nope" })
      assert_raises(Client::SessionNotFound) { @client.get_session("gone") }
    end

    test "delete_session returns true on 204 and tolerates 404" do
      stub_request(:delete, "#{BASE}/sess-1").to_return(status: 204)
      stub_json(:delete, "#{BASE}/gone", status: 404, body: { detail: "nope" })
      assert_equal true, @client.delete_session("sess-1")
      assert_equal true, @client.delete_session("gone")
    end

    test "delete_session raises ServerError on other failures" do
      stub_request(:delete, "#{BASE}/sess-1").to_return(status: 500, body: "boom")
      assert_raises(Client::ServerError) { @client.delete_session("sess-1") }
    end

    test "estimate_in_session sends multipart fields and one part per attachment" do
      captured = nil
      stub = stub_request(:post, "#{BASE}/sess-1/estimate")
        .with { |req| captured = req; true }
        .to_return(status: 200, headers: JSON_HEADERS,
                   body: session_estimation_payload(attachments: [ "a.txt", "b.md" ]).to_json)

      files = [ upload("a.txt", "alpha", "text/plain"), nil, upload("b.md", "# beta", "text/markdown") ]
      payload = @client.estimate_in_session("sess-1", session_request(tier: "pm"), attachments: files)

      assert_requested stub
      assert_match %r{\Amultipart/form-data; boundary=}, captured.headers["Content-Type"]
      body = captured.body
      assert_includes body, 'name="transcript"'
      assert_includes body, "Add a mobile app to the portal."
      assert_includes body, 'name="hourly_rate"'
      assert_includes body, 'name="tier"'
      assert_equal 2, body.scan(/name="attachments"; filename="/).size
      assert_includes body, 'filename="a.txt"'
      assert_includes body, 'filename="b.md"'
      assert_equal "sess-1", payload["session_id"]
      assert_equal %w[a.txt b.md], payload["attachments"]
    end

    test "estimate_in_session omits tier and attachments when absent" do
      captured = nil
      stub_request(:post, "#{BASE}/sess-1/estimate")
        .with { |req| captured = req; true }
        .to_return(status: 200, headers: JSON_HEADERS, body: session_estimation_payload.to_json)

      @client.estimate_in_session("sess-1", session_request)

      assert_not_includes captured.body, 'name="tier"'
      assert_not_includes captured.body, 'name="attachments"'
    end

    test "estimate_in_session raises ArgumentError for an invalid request and makes no call" do
      assert_raises(ArgumentError) { @client.estimate_in_session("sess-1", session_request(transcript: "")) }
      assert_not_requested :post, "#{BASE}/sess-1/estimate"
    end

    test "estimate_in_session maps 404 to SessionNotFound" do
      stub_json(:post, "#{BASE}/sess-1/estimate", status: 404, body: { detail: "expired" })
      assert_raises(Client::SessionNotFound) { @client.estimate_in_session("sess-1", session_request) }
    end

    %w[moderation prompt_injection pii].each do |reason|
      test "estimate_in_session 400 #{reason} raises GuardrailViolation" do
        stub_json(:post, "#{BASE}/sess-1/estimate", status: 400, body: { reason: reason, message: "Safe #{reason}." })

        error = assert_raises(Client::GuardrailViolation) { @client.estimate_in_session("sess-1", session_request) }
        assert_equal reason, error.reason
        assert_equal "Safe #{reason}.", error.message
      end
    end

    test "estimate_in_session other 400 raises InvalidRequest" do
      stub_json(:post, "#{BASE}/sess-1/estimate", status: 400, body: { reason: "weird" })
      assert_raises(Client::InvalidRequest) { @client.estimate_in_session("sess-1", session_request) }
    end

    { 413 => "too large", 415 => "unsupported", 422 => "invalid" }.each do |status, _|
      test "estimate_in_session #{status} raises InvalidRequest with a safe message" do
        stub_json(:post, "#{BASE}/sess-1/estimate", status: status, body: { detail: "SECRET internals" })

        error = assert_raises(Client::InvalidRequest) { @client.estimate_in_session("sess-1", session_request) }
        assert_not_includes error.message, "SECRET"
        assert_not_empty error.message
      end
    end

    test "estimate_in_session 502 and unexpected statuses raise ServerError" do
      stub_json(:post, "#{BASE}/sess-1/estimate", status: 502, body: { detail: "SECRET" })
      error = assert_raises(Client::ServerError) { @client.estimate_in_session("sess-1", session_request) }
      assert_not_includes error.message, "SECRET"

      stub_request(:post, "#{BASE}/sess-1/estimate").to_return(status: 500, body: "Traceback SECRET")
      error = assert_raises(Client::ServerError) { @client.estimate_in_session("sess-1", session_request) }
      assert_equal "unexpected status 500", error.message
    end
  end
end
