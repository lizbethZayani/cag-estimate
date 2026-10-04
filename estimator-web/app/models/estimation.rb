class Estimation < ApplicationRecord
  validates :transcription, :hourly_rate, presence: true

  def to_response
    EstimationResponse.from_hash(response_payload)
  end

  def transcription_preview(limit: 80)
    transcription.to_s.truncate(limit)
  end
end
