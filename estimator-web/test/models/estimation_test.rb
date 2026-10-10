require "test_helper"

class EstimationTest < ActiveSupport::TestCase
  def build(overrides = {})
    Estimation.new({
      transcription: "We need a customer portal with invoices and payments.",
      hourly_rate: 50,
      response_payload: estimation_payload,
      prompt_version: "v1",
      cached: false
    }.merge(overrides))
  end

  test "is valid with required fields" do
    assert build.valid?
  end

  test "requires transcription and hourly_rate" do
    assert_not build(transcription: "").valid?
    assert_not build(hourly_rate: nil).valid?
  end

  test "to_response rebuilds the EstimationResponse from the payload" do
    response = build.to_response
    assert_kind_of EstimationResponse, response
    assert_equal 298, response.result.summary.total_hours
  end

  test "transcription_preview truncates" do
    assert_equal 20, build(transcription: "x" * 100).transcription_preview(limit: 20).length
    assert_equal 80, build(transcription: "x" * 100).transcription_preview.length
  end

  test "chat_session is optional and can be assigned" do
    assert build.valid?
    session = ChatSession.create!(remote_session_id: "s-1")
    estimation = build(chat_session: session).tap(&:save!)
    assert_equal session, estimation.reload.chat_session
  end

  test "persists the payload as jsonb" do
    estimation = build.tap(&:save!)
    assert_equal 298, Estimation.find(estimation.id).response_payload.dig("result", "summary", "total_hours")
  end
end
