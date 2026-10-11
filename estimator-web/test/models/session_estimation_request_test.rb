require "test_helper"

class SessionEstimationRequestTest < ActiveSupport::TestCase
  def build(overrides = {})
    SessionEstimationRequest.new({ transcript: "Add a mobile app to the portal." }.merge(overrides))
  end

  test "is valid with defaults" do
    request = build
    assert request.valid?
    assert_equal 40, request.hourly_rate
  end

  test "requires a transcript of at most 80000 chars" do
    assert_not build(transcript: "").valid?
    assert_not build(transcript: "x" * 80_001).valid?
    assert build(transcript: "x" * 80_000).valid?
  end

  test "hourly rate must be an integer within 30..150" do
    assert_not build(hourly_rate: 29).valid?
    assert_not build(hourly_rate: 151).valid?
    assert_not build(hourly_rate: nil).valid?
    assert build(hourly_rate: 30).valid?
    assert build(hourly_rate: 150).valid?
  end

  test "tier is optional but restricted" do
    assert build(tier: "").valid?
    assert build(tier: nil).valid?
    %w[executive pm developer].each { |t| assert build(tier: t).valid? }
    assert_not build(tier: "ceo").valid?
  end

  test "to_form_fields omits blank values" do
    assert_equal({ transcript: "Add a mobile app to the portal.", hourly_rate: 40 }, build.to_form_fields)
    assert_equal "pm", build(tier: "pm").to_form_fields[:tier]
  end
end
