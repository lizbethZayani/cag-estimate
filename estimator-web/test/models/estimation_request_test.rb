require "test_helper"

class EstimationRequestTest < ActiveSupport::TestCase
  def build(overrides = {})
    EstimationRequest.new({ transcription: "A" * 30, hourly_rate: 40 }.merge(overrides))
  end

  test "is valid with a transcription and a rate" do
    assert build.valid?
  end

  test "defaults hourly_rate to 40" do
    assert_equal 40, EstimationRequest.new(transcription: "A" * 30).hourly_rate
  end

  test "requires a transcription" do
    assert_not build(transcription: "").valid?
  end

  test "rejects too short and too long transcriptions" do
    assert_not build(transcription: "short").valid?
    assert_not build(transcription: "A" * 80_001).valid?
    assert build(transcription: "A" * 80_000).valid?
  end

  test "hourly_rate must be an integer within 30..150" do
    assert_not build(hourly_rate: 29).valid?
    assert_not build(hourly_rate: 151).valid?
    assert_not build(hourly_rate: nil).valid?
    assert build(hourly_rate: 30).valid?
    assert build(hourly_rate: 150).valid?
  end

  test "to_payload returns the API request body" do
    assert_equal({ transcription: "A" * 30, hourly_rate: 55 }, build(hourly_rate: 55).to_payload)
  end
end
