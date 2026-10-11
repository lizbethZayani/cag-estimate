require "test_helper"

class ChatSessionTest < ActiveSupport::TestCase
  def build(overrides = {})
    ChatSession.new({ remote_session_id: "abc-123" }.merge(overrides))
  end

  test "is valid with a remote session id" do
    assert build.valid?
  end

  test "requires a unique remote session id" do
    assert_not build(remote_session_id: "").valid?
    build.save!
    assert_not build.valid?
  end

  test "defaults to empty metadata and zero turns" do
    session = build.tap(&:save!).reload
    assert_equal({}, session.latest_metadata)
    assert_equal 0, session.turn_count
  end

  test "latest_metadata_hash returns string keys with safe defaults" do
    meta = build.latest_metadata_hash
    assert_nil meta["project_name"]
    assert_nil meta["assumed_team_size"]
    assert_equal [], meta["mentioned_technologies"]
    assert_nil meta["agreed_scope"]
  end

  test "latest_metadata_hash keeps stored values" do
    session = build(latest_metadata: { "project_name" => "Portal", "mentioned_technologies" => [ "Rails" ] })
    meta = session.latest_metadata_hash
    assert_equal "Portal", meta["project_name"]
    assert_equal [ "Rails" ], meta["mentioned_technologies"]
  end

  test "title uses the project name when present" do
    assert_equal "Portal", build(latest_metadata: { "project_name" => "Portal" }).title
  end

  test "title falls back to a generic label" do
    assert_equal "Untitled conversation", build.title
  end

  test "has many estimations and nullifies them on destroy" do
    session = build.tap(&:save!)
    estimation = Estimation.create!(
      transcription: "We need a portal.", hourly_rate: 50,
      response_payload: estimation_payload, chat_session: session
    )
    assert_equal [ estimation ], session.estimations.to_a

    session.destroy!
    assert_nil estimation.reload.chat_session_id
  end
end
