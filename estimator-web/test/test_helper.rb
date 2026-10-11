ENV["RAILS_ENV"] ||= "test"
require_relative "../config/environment"
require "rails/test_help"
require "webmock/minitest"

WebMock.disable_net_connect!

module ActiveSupport
  class TestCase
    # Run tests in parallel with specified workers
    parallelize(workers: :number_of_processors)

    # Setup all fixtures in test/fixtures/*.yml for all tests in alphabetical order.
    fixtures :all

    # Real response shape captured from POST /api/v1/estimate.
    def estimation_payload(cached: false)
      ::JSON.parse(file_fixture("estimation_response.json").read).merge("cached" => cached)
    end

    # Shape of POST /api/v1/sessions/{id}/estimate (result + session fields).
    def session_estimation_payload(session_id: "sess-1", attachments: [])
      estimation_payload.merge(
        "prompt_version" => "v2",
        "session_id" => session_id,
        "project_metadata" => {
          "project_name" => "Customer Portal", "assumed_team_size" => 4,
          "mentioned_technologies" => [ "Rails" ], "agreed_scope" => "Invoices"
        },
        "turns" => 1,
        "attachments" => attachments,
        "resolved_tier" => "developer",
        "tier_rule" => "default"
      )
    end
  end
end
