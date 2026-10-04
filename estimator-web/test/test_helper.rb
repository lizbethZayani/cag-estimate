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
  end
end
