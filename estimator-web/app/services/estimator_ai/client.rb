require "faraday"

module EstimatorAi
  class Client
    Error          = Class.new(StandardError)
    InvalidRequest = Class.new(Error)
    ServerError    = Class.new(Error)

    # Input rejected by the API guardrails. `message` is the API's safe,
    # user-facing text; `reason` is one of GUARDRAIL_REASONS.
    class GuardrailViolation < Error
      attr_reader :reason

      def initialize(message = nil, reason: nil)
        super(message)
        @reason = reason
      end
    end

    GUARDRAIL_REASONS = %w[moderation prompt_injection pii].freeze

    def initialize(base_url: Rails.application.config.estimator_ai.base_url,
                   timeout:  Rails.application.config.estimator_ai.timeout)
      @conn = Faraday.new(url: base_url) do |f|
        f.request  :json
        f.response :json
        f.options.timeout = timeout
        f.adapter Faraday.default_adapter
      end
    end

    def estimate(request)
      raise ArgumentError, "request must be valid" unless request.valid?

      response = @conn.post("/api/v1/estimate", request.to_payload)
      body = response.body

      case response.status
      when 200
        body
      when 400
        raise_for_bad_request(body)
      when 422
        raise InvalidRequest, validation_message(body)
      when 502
        raise ServerError, "Upstream LLM call failed"
      else
        raise ServerError, "unexpected status #{response.status}"
      end
    end

    private

    def raise_for_bad_request(body)
      reason = body.is_a?(Hash) ? body["reason"] : nil
      if GUARDRAIL_REASONS.include?(reason)
        message = body["message"].presence || "Input rejected (#{reason})"
        raise GuardrailViolation.new(message, reason: reason)
      end
      raise InvalidRequest, "The API rejected the request"
    end

    def validation_message(body)
      detail = body.is_a?(Hash) ? body["detail"] : nil
      return "The API rejected the request" unless detail.is_a?(Array) && detail.any?

      detail.map do |item|
        next item.to_s unless item.is_a?(Hash)
        "#{Array(item["loc"]).join(".")}: #{item["msg"]}"
      end.join("; ")
    end
  end
end
