class EstimationRequest
  include ActiveModel::Model
  include ActiveModel::Attributes

  # Bounds accepted by the API's output validators (ProjectSummary.hourly_rate).
  MIN_HOURLY_RATE = 30
  MAX_HOURLY_RATE = 150

  attribute :transcription, :string
  attribute :hourly_rate,   :integer, default: 40

  validates :transcription, presence: true, length: { in: 20..80_000 }
  validates :hourly_rate, presence: true,
            numericality: { only_integer: true,
                            greater_than_or_equal_to: MIN_HOURLY_RATE,
                            less_than_or_equal_to: MAX_HOURLY_RATE }

  def to_payload
    { transcription: transcription, hourly_rate: hourly_rate }
  end
end
