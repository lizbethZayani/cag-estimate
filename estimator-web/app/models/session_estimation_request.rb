# Form object for POST /api/v1/sessions/{id}/estimate (multipart).
class SessionEstimationRequest
  include ActiveModel::Model
  include ActiveModel::Attributes

  TIERS = %w[executive pm developer].freeze
  TRANSCRIPT_MAX = 80_000

  attribute :transcript,  :string
  attribute :hourly_rate, :integer, default: 40
  attribute :tier,        :string

  validates :transcript, presence: true, length: { maximum: TRANSCRIPT_MAX }
  validates :hourly_rate, presence: true,
            numericality: { only_integer: true,
                            greater_than_or_equal_to: EstimationRequest::MIN_HOURLY_RATE,
                            less_than_or_equal_to: EstimationRequest::MAX_HOURLY_RATE }
  validates :tier, inclusion: { in: TIERS }, allow_blank: true

  def to_form_fields
    { transcript: transcript, hourly_rate: hourly_rate, tier: tier }.compact_blank
  end
end
